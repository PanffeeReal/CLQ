'''
Codes are adapted from https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/fcos_head.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Scale, normal_init
from mmcv.runner import force_fp32

from mmdet.core import distance2bbox, multi_apply, multiclass_nms, reduce_mean,bbox_overlaps
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead
import numpy as np
from mmcv.ops import DeformConv2d

INF = 1e8

def weighted_binary_cross_entropy(pred, label, weight, avg_factor=None):
    if pred.dim() != label.dim():
        label, weight = _expand_binary_labels(label, weight, pred.size(-1))
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.)
    return F.binary_cross_entropy_with_logits(
        pred, label.float(), weight.float(),
        reduction='sum')[None] / avg_factor
def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(
        label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights
#add feature alignment module --> use the whole CLQ head: lpf 0613
class FeatureAlignment(nn.Module):
    """Feature Adaption Module.
    Feature Adaption Module is implemented based on DCN v1.
    It uses anchor shape prediction rather than feature map to
    predict offsets of deformable conv layer.
    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output feature map.
        kernel_size (int): Deformable conv kernel size.
        deformable_groups (int): Deformable conv group size.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deformable_groups=4):
        super(FeatureAlignment, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(
            4, deformable_groups * offset_channels, 1, bias=False)
        self.conv_adaption = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        normal_init(self.conv_offset, std=0.1)
        normal_init(self.conv_adaption, std=0.01)

    def forward(self, x, shape):
        offset = self.conv_offset(shape)
        x = self.relu(self.conv_adaption(x, offset))
        return x

@HEADS.register_module()
class PseudoBox_Head_lqe(AnchorFreeHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),(512, INF)),
                 #  origion
                 #((1, 64), (32, 128), (64, 256), (128, 512), (256, 2048)) ## lpf add 0731 ; like fovea
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 piou = False,
                 piou_thr = 0.4,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 **kwargs):
        self.regress_ranges = regress_ranges
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        self.piou = piou
        self.piou_thr = piou_thr
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            **kwargs)
        self.loss_centerness = build_loss(loss_centerness)

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        #lpf add 20240416
        self.lqe_layer = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        #lpf add 20240620
        self.deformable_groups = 4
        self.use_feature_alignment = True
        if self.use_feature_alignment:
            self.feature_alignment = FeatureAlignment(
                self.feat_channels,
                self.feat_channels,
                kernel_size=3,
                deformable_groups=self.deformable_groups)


    def init_weights(self):
        """Initialize weights of the head."""
        super().init_weights()
        normal_init(self.conv_centerness, std=0.01)
        #lpf add 20240416
        normal_init(self.lqe_layer, std=0.01)
        #lpf add 20240620
        if self.use_feature_alignment:
            self.feature_alignment.init_weights()
       
    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                centernesses (list[Tensor]): centerness for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides)

    def forward_single(self, x, scale, stride):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and centerness \ 
                predictions of input feature maps.
        """
 #       cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)
  
        cls_feat = x
        reg_feat = x

#        for cls_layer in self.cls_convs:
#            cls_feat = cls_layer(cls_feat)
#        cls_score = self.conv_cls(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)

        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = F.relu(bbox_pred)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        # lpf add 20240620
        if self.use_feature_alignment:
            if self.norm_on_bbox:
                if not self.training:
                    bbox_pred_dcn= bbox_pred/stride            
                else:
                    bbox_pred_dcn=bbox_pred
            else:
                bbox_pred_dcn=bbox_pred
            cls_feat=self.feature_alignment(cls_feat,bbox_pred_dcn) #lpf 0323 add for feature adaption
            reg_feat=self.feature_alignment(reg_feat,bbox_pred_dcn) # lpf 20240111 add

        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        cls_score = self.conv_cls(cls_feat)
        #lpf add 20240416  ablation 1； lqe
        quality_score = self.lqe_layer(reg_feat)
        cls_score = (cls_score.sigmoid()) * ((quality_score.sigmoid()).pow(0.75))  # lpf 0.8/0.2 1/0.3 0.2  1.0/0.75

        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)

        return cls_score, bbox_pred, centerness,quality_score

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             quality_score,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses) == len(quality_score) ##lpf add 20240416 quality score
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        # lpf revise : add output: gt_inds  :  return  gt_inds
        labels, bbox_targets, gt_inds= self.get_targets(all_level_points, gt_bboxes,
                                                gt_labels)
        flatten_gt_inds = torch.cat(gt_inds) # lpf add

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)

        #lpf add 20240416
        flatten_quality_score = [
            quality_score_single.permute(0, 2, 3, 1).reshape(-1)
            for quality_score_single in quality_score
        ]
        flatten_quality_score=torch.cat(flatten_quality_score)
        lqe_target = flatten_labels.new_zeros(flatten_labels.shape).float()

        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes

        if self.piou:
            pos_inds_tem = ((flatten_labels >= 0) & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
            # print('pos_inds_tem',pos_inds_tem.shape,pos_inds_tem.max())
            # print('pos_inds_tmp',len(pos_inds_tem))
            pos_bbox_targets_tem = flatten_bbox_targets[pos_inds_tem].detach()
            ## lpf add 0721
            pos_classcore_tem,_ = flatten_cls_scores.detach()[pos_inds_tem].max(dim=1) #have used sigmoid in forward_single()
            ## check class_labels
            ##flatten_cls_scores[pos_inds_tem].max(dim=1) == flatten_cls_scores[pos_inds_tem][:,flatten_labels]

            pos_classcore_threshold=torch.tensor(0.4)
            left = pos_bbox_targets_tem[:, 0]
            right = pos_bbox_targets_tem[:, 2]
            top = pos_bbox_targets_tem[:, 1]
            bottom = pos_bbox_targets_tem[:, 3]
            inter_left = left.clone()
            inter_right = right.clone()
            inter_top = top.clone()
            inter_bottom = bottom.clone()
            #---------------------------------
            # half_w = (left+right)/2
            # half_h = (top+bottom)/2
            # for i in range(len(left)):
            #     if half_w[i]<left[i]:
            #         inter_left[i] = half_w[i]
            #     if half_w[i]<right[i]:
            #         inter_right[i] = half_w[i]
            #     if half_h[i]<top[i]:
            #         inter_top[i] = half_h[i]
            #     if half_h[i]<bottom[i]:
            #         inter_bottom[i] = half_h[i]
            area_u = (left+right)*(top+bottom) # #
            # area_i = (inter_left+inter_right)*(inter_top+inter_bottom)
            # iou_target = area_i/(area_u+area_u-area_i)
            # print('iou_target', iou_target.shape)
            # lpf change pseudo-box --> predict box   ablation 2:regressed box as well as use cls_score & iou
            pos_bbox_preds_tem = flatten_bbox_preds[pos_inds_tem]
            left_preds = pos_bbox_preds_tem[:, 0]
            right_preds = pos_bbox_preds_tem[:, 2]
            top_preds = pos_bbox_preds_tem[:, 1]
            bottom_preds = pos_bbox_preds_tem[:, 3]
            for i in range(len(left)):
                if left_preds[i] < left[i]:
                    inter_left[i] = left_preds[i]
                if right_preds[i] < right[i]:
                    inter_right[i] = right_preds[i]
                if top_preds[i] < top[i]:
                    inter_top[i] = top_preds[i]
                if bottom_preds[i] < bottom[i]:
                    inter_bottom[i] = bottom_preds[i]
            area_p = (left_preds+right_preds)*(top_preds+bottom_preds)
            area_i = (inter_left + inter_right) * (inter_top + inter_bottom)
            iou_target = area_i / (area_p + area_u - area_i)

            # pos_inds = pos_inds_tem[iou_target>self.piou_thr]

            # ##lpf add 0722 : set a new para to represent alignment of cls and loc
            #if joint_feature_selection: need to consider the threshold self.piou_thr and the type of the combination align_score=iou_alpha*cls_beita >= threshold
            # align_score=iou_target*(pos_labels_tem.pow(0.1))
            # pos_inds = pos_inds_tem[align_score > self.piou_thr]
            ## lpf add 0722 seperate considering loc and cls
            #else: set loc's threshold

            # pos_inds_temp: (num_pos_tem,)
            # pos_inds: (num_pos,)
            positives_bool=(iou_target > self.piou_thr)#& (pos_classcore_tem>pos_classcore_threshold)  #further select pos_inds
            pos_inds = pos_inds_tem[positives_bool]
            # pos_inds_ignore = pos_inds_tem[align_score<=self.piou_thr] #lpf change iou_target --> align_score
            pos_inds_ignore = pos_inds_tem[(iou_target <= self.piou_thr)] # |(pos_classcore_tem<=pos_classcore_threshold)
            # print('1_num_postives',len(pos_inds),len(pos_inds_ignore),len(pos_inds)+len(pos_inds_ignore))
            # print('before dynamic la',len(pos_inds))
            # print('flatten_gt_inds2222', flatten_gt_inds[pos_inds]==)

            #lpf add 20240415
            topk = 100
            #alpha=0.3
            assert iou_target.shape == pos_classcore_tem.shape  # the shape of iou_target == pos_inds.shape

            # pos_inds.shape == positives_bool.nonzero().shape ;; but the pos_inds.max() != positives_bool.nonzero().shape --> so use the positive_bool
            align_score=iou_target[positives_bool]#.pow(alpha) * pos_classcore_tem[positives_bool].pow(1-alpha)   #ablation 3: if consider cls_score  
            # lpf add 0709 : normlization align_score            
            align_score=align_score/align_score.max()
                
            # align_score=pos_classcore_tem[positives_bool]
            num_gts = sum(map(len, gt_labels))  # total number of gt in the batch
            # print('checkpoint1',gt_labels)
            # print('checkpoint2',num_gts)
            device = flatten_labels.device
            label_sequence = torch.arange(num_gts,device=device)#  if add device --> empty label_sequence
            pos_inds_ = []


            #ablation 4: dynamic label assignment
            #calculate pos_inds w.r.t each gt
            # pos_inds_tmp <-- flatten_labels(positives).nonzeros()
            # pos_inds <--> align_score/iou_target/pos_classcore_tem
            for i, l in enumerate(label_sequence):
                # for each gt, select the corresponding pos_inds
                # pos_inds_[i] <--> align_score_ / mask_for_score
                mask_for_score = (flatten_gt_inds[pos_inds] == l)
                # print('i,l',i,l)
                # print('flatten_gt_inds',len(mask_for_score.nonzero()))
                # print('flatten_gt_inds_iii',flatten_gt_inds[pos_inds][mask_for_score])
                if mask_for_score.any():  #maybe some positives have been filter out in the forementioned steps
                    align_score_ = align_score[mask_for_score]
                    # print('check 00000', len(align_score_))
                    # pos_inds_i = pos_inds[mask_for_score.nonzero().flatten()]
                    pos_inds_i = pos_inds[mask_for_score]
                    # print('determine11111',flatten_labels[pos_inds_i])
                    # print('determine1111111111111', align_score_)
                    # do the dynamic label assign
                    if topk >= align_score_.shape[0]:
                        # print('1111111111111')
                        sorted_align_score,topk_indices=torch.topk(align_score_,k=align_score_.shape[0],dim=0)
                        # print('determine222222',align_score_[topk_indices]==sorted_align_score)
                    else:
                        # print('222222222222222')
                        # sorted_align_score, topk_indices = torch.topk(align_score, k=topk, dim=0)
                        sorted_align_score, topk_indices = torch.sort(align_score_, dim=0, descending=True)  ### align_score  <--> pos_inds_i : corresponse
                        sorted_align_score = sorted_align_score[:topk]
                        pos_inds_ignore = torch.cat((pos_inds_ignore, pos_inds_i[topk_indices[topk:]]))
                        pos_inds_i = pos_inds_i[topk_indices[:topk]]   ##### sorted_align_score   <-->  this new pos_inds_i : corresponse : length is also the same
                        # print('aiaiaiaiaiai',flatten_gt_inds[pos_inds_i]==l,sorted_align_score.min(),sorted_align_score.max())
                    score_gap=0.4
                    score_division=150
                    # assert topk==sorted_align_score.shape[0]
                    score_diff = sorted_align_score - torch.roll(sorted_align_score,-1,dims=0)

                    top_m=min(topk,align_score_.shape[0])
                    min_num = 2 # ensure at least 10 positives will be selected for each gt; except for those whose positives are origionally less than min_num
                    if len(score_diff)>=min_num:
                        for i in range(score_diff.shape[0]-2):
                            # division mode
                            if score_diff[i+1]/score_diff[i] > score_division:
                                # print('44444444444')
                                # lpf add 20240616 : ensure at least min_num samples will be selected:
                                #if min_num >= i+2:
                                #    top_m=min_num
                                #else:
                                top_m=i+2  # renew topm : for example: if score_diff[0+1]/score_diff[0] > score_division, keep the 2nd samples: sorted_align_score[:0+2]
                                # print('top_m',top_m)
                                break
                            else:
                                pass
                            # diff mode:
                            # if score_diff[i] > score_gap:
                            #     top_m=i+1
                            #     print('top_m', top_m)
                            #     break
                            # else:
                            #     pass
                        if top_m < min(topk,align_score_.shape[0]):
                            # print('3333333333333')
                            pos_inds_ignore = torch.cat((pos_inds_ignore,pos_inds_i[top_m:]))
                            pos_inds_i = pos_inds_i[:top_m]
                    pos_inds_.extend(pos_inds_i)
                    # print('i, num_postives',i,len(pos_inds_))
                ## end gt_i searching
            pos_inds = torch.tensor(pos_inds_)  #lpf must be long : for the following usage
            # print('dynamic la', len(pos_inds))
            flatten_labels[pos_inds_ignore] = bg_class_ind
        else:
            pos_inds = ((flatten_labels >= 0) & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        # print('2_num_postives', len(pos_inds), len(pos_inds_ignore),len(pos_inds)+len(pos_inds_ignore))
        # print('sssssssssssssssssss',pos_inds.shape)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        # print(type(pos_inds))
        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        # print('pos_inds',pos_inds)
        if len(pos_inds) > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_centerness_targets = self.centerness_target(pos_bbox_targets)
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            #### lpf:check if this step is right:   need need to check
            # print(iou_target[pos_inds] == bbox_overlaps(
            #      pos_decoded_bbox_preds.detach(),
            #      pos_decoded_target_preds,
            #      is_aligned=True)
            # can't use iou_target[pos_inds] !! because pos_inds corresponds to (num_points,), but iou_target.shape = len(pos_inds)


            #lpf add 20240416 quality score
            lqe_target[pos_inds]=bbox_overlaps(
                 pos_decoded_bbox_preds.detach(),
                 pos_decoded_target_preds,
                 is_aligned=True)


            #lpf add : weight function
            k=torch.tensor(15.0)
            cls_consistency_new_pos_inds,_=flatten_cls_scores.detach()[pos_inds].max(1)
            cls_loc_consistency=abs(lqe_target[pos_inds]-cls_consistency_new_pos_inds)

            # loc_weights=((k*cls_loc_consistency).exp()-(k).exp())/(1-(k).exp())

            ####### lpf: use function or directly use cls_consistency_new_pos_inds (like GFLV2: now it contains both cls_score and lqe_score)
            #!!!!!!!!!!!!!!compute and analyze the distribution of cls_loc_consistency :: --> design the suitable function and parameter
            # loc_weights=1./(1.0+k*cls_loc_consistency) #.pow(0.5)              #### function 1 next
            # loc_weights=cls_loc_consistency                                    #### directly use function 2
            # loc_weights = cls_consistency_new_pos_inds                         # directly use like GFLv2   function 3
            # loc_weights=cls_loc_consistency * cls_consistency_new_pos_inds     #### function 4: cls_score & iou
            # loc_weights=cls_loc_consistency * iou_target[pos_inds].pow(0.3)             #### function 5: only iou
            loc_weights=(1./(1.0+k*cls_loc_consistency))*(((lqe_target[pos_inds])*cls_consistency_new_pos_inds).pow(0.5))

            loc_weights = loc_weights/loc_weights.max()



            # assert len(pos_centerness_targets)==len(loc_weights)   # ablation 5
            pos_centerness_targets_1=pos_centerness_targets * loc_weights  # lpf revise: norm loc_weights --> loc_weights./loc_weights.max()

            # centerness weighted iou loss
            centerness_denorm = max(
                reduce_mean(pos_centerness_targets_1.sum().detach()), 1e-6)

            ## lqe_branch's loss calculation
            # print(type(lqe_target),type(iou_target),type(lqe_target[pos_inds]),type(iou_target[pos_inds]))
            loss_iou_weight=1.0
            # lqe_weight = lqe_target.new_ones(pos_inds.shape)
            lqe_loss_weight_switch=True   ## ablation 6: if weight lqe_loss and how to weight is : using which function
            if lqe_loss_weight_switch:
                loss_iou = loss_iou_weight*weighted_binary_cross_entropy(flatten_quality_score[pos_inds], lqe_target[pos_inds], weight=pos_centerness_targets_1,
                                                                     avg_factor=centerness_denorm)
            else:
                loss_iou = loss_iou_weight * weighted_binary_cross_entropy(flatten_quality_score[pos_inds],
                                                                           lqe_target[pos_inds],
                                                                           weight=torch.tensor(1),
                                                                           avg_factor=num_pos)

            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets_1,   ##lpf will add : weight function to replace centerness by considering cls and loc
                avg_factor=centerness_denorm)
            loss_centerness =  self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)   #no_centerness 0716
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()
            loss_iou = pos_bbox_preds.sum()*0
        #lpf add 0905
        # print(flatten_labels.dtype,loc_weights.dtype)
        # cls_weights=torch.ones(flatten_labels.size()).cuda().detach()
        # # cls_weights[pos_inds]=loc_weights.cuda().detach()
        # cls_weights[pos_inds]=iou_target[(iou_target > self.piou_thr) & (pos_labels_tem>0.5)] * loc_weights.cuda().detach()
        # cls_weights[pos_inds]=cls_weights[pos_inds]/(cls_weights[pos_inds].max())
        # num_pos=cls_weights[pos_inds].sum()
        # loss_cls = self.loss_cls(
        #     flatten_cls_scores, flatten_labels,weight=cls_weights, avg_factor=num_pos)
        loss_cls = self.loss_cls(
            flatten_cls_scores, (flatten_labels,lqe_target), avg_factor=num_pos)

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness,
            loss_iou=loss_iou)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   lqe_scores, # lpf add for lqe branch's usage
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(
                cls_score_list, bbox_pred_list, centerness_pred_list,
                mlvl_points, img_shape, scale_factor, cfg, rescale, with_nms)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           centernesses,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points * 4, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, centerness, points in zip(
                cls_scores, bbox_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels)#.sigmoid()
            #lpf revise in 20240416 , comment sigmoid() as in forward_single(). have used sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms(
                mlvl_bboxes,
                mlvl_scores,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                score_factors=mlvl_centerness)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores, mlvl_centerness
    # def get_bboxes(self,
    #                cls_scores,
    #                bbox_preds,
    #                centernesses,
    #                img_metas,
    #                cfg=None,
    #                rescale=False,
    #                with_nms=True):
    #     """Transform network output for a batch into bbox predictions.
    #
    #     Args:
    #         cls_scores (list[Tensor]): Box scores for each scale level
    #             with shape (N, num_points * num_classes, H, W).
    #         bbox_preds (list[Tensor]): Box energies / deltas for each scale
    #             level with shape (N, num_points * 4, H, W).
    #         centernesses (list[Tensor]): Centerness for each scale level with
    #             shape (N, num_points * 1, H, W).
    #         img_metas (list[dict]): Meta information of each image, e.g.,
    #             image size, scaling factor, etc.
    #         cfg (mmcv.Config | None): Test / postprocessing configuration,
    #             if None, test_cfg would be used. Default: None.
    #         rescale (bool): If True, return boxes in original image space.
    #             Default: False.
    #         with_nms (bool): If True, do nms before return boxes.
    #             Default: True.
    #
    #     Returns:
    #         list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
    #             The first item is an (n, 5) tensor, where 5 represent
    #             (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
    #             The shape of the second tensor in the tuple is (n,), and
    #             each element represents the class label of the corresponding
    #             box.
    #     """
    #     assert len(cls_scores) == len(bbox_preds)
    #     num_levels = len(cls_scores)
    #
    #     featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
    #     mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
    #                                   bbox_preds[0].device)
    #
    #     cls_score_list = [cls_scores[i].detach() for i in range(num_levels)]
    #     bbox_pred_list = [bbox_preds[i].detach() for i in range(num_levels)]
    #     centerness_pred_list = [
    #         centernesses[i].detach() for i in range(num_levels)
    #     ]
    #     if torch.onnx.is_in_onnx_export():
    #         assert len(
    #             img_metas
    #         ) == 1, 'Only support one input image while in exporting to ONNX'
    #         img_shapes = img_metas[0]['img_shape_for_onnx']
    #     else:
    #         img_shapes = [
    #             img_metas[i]['img_shape']
    #             for i in range(cls_scores[0].shape[0])
    #         ]
    #     scale_factors = [
    #         img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])
    #     ]
    #     result_list = self._get_bboxes(cls_score_list, bbox_pred_list,
    #                                    centerness_pred_list, mlvl_points,
    #                                    img_shapes, scale_factors, cfg, rescale,
    #                                    with_nms)
    #     return result_list
    #
    # def _get_bboxes(self,
    #                 cls_scores,
    #                 bbox_preds,
    #                 centernesses,
    #                 mlvl_points,
    #                 img_shapes,
    #                 scale_factors,
    #                 cfg,
    #                 rescale=False,
    #                 with_nms=True):
    #     """Transform outputs for a single batch item into bbox predictions.
    #
    #     Args:
    #         cls_scores (list[Tensor]): Box scores for a single scale level
    #             with shape (N, num_points * num_classes, H, W).
    #         bbox_preds (list[Tensor]): Box energies / deltas for a single scale
    #             level with shape (N, num_points * 4, H, W).
    #         centernesses (list[Tensor]): Centerness for a single scale level
    #             with shape (N, num_points * 4, H, W).
    #         mlvl_points (list[Tensor]): Box reference for a single scale level
    #             with shape (num_total_points, 4).
    #         img_shapes (list[tuple[int]]): Shape of the input image,
    #             list[(height, width, 3)].
    #         scale_factors (list[ndarray]): Scale factor of the image arrange as
    #             (w_scale, h_scale, w_scale, h_scale).
    #         cfg (mmcv.Config | None): Test / postprocessing configuration,
    #             if None, test_cfg would be used.
    #         rescale (bool): If True, return boxes in original image space.
    #             Default: False.
    #         with_nms (bool): If True, do nms before return boxes.
    #             Default: True.
    #
    #     Returns:
    #         tuple(Tensor):
    #             det_bboxes (Tensor): BBox predictions in shape (n, 5), where
    #                 the first 4 columns are bounding box positions
    #                 (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
    #                 between 0 and 1.
    #             det_labels (Tensor): A (n,) tensor where each item is the
    #                 predicted class label of the corresponding box.
    #     """
    #     cfg = self.test_cfg if cfg is None else cfg
    #     assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
    #     device = cls_scores[0].device
    #     batch_size = cls_scores[0].shape[0]
    #     # convert to tensor to keep tracing
    #     nms_pre_tensor = torch.tensor(
    #         cfg.get('nms_pre', -1), device=device, dtype=torch.long)
    #     mlvl_bboxes = []
    #     mlvl_scores = []
    #     mlvl_centerness = []
    #     for cls_score, bbox_pred, centerness, points in zip(
    #             cls_scores, bbox_preds, centernesses, mlvl_points):
    #         assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
    #         scores = cls_score.permute(0, 2, 3, 1).reshape(
    #             batch_size, -1, self.cls_out_channels).sigmoid()
    #         centerness = centerness.permute(0, 2, 3,
    #                                         1).reshape(batch_size,
    #                                                    -1).sigmoid()
    #
    #         bbox_pred = bbox_pred.permute(0, 2, 3,
    #                                       1).reshape(batch_size, -1, 4)
    #         # Always keep topk op for dynamic input in onnx
    #         if nms_pre_tensor > 0 and (torch.onnx.is_in_onnx_export()
    #                                    or scores.shape[-2] > nms_pre_tensor):
    #             from torch import _shape_as_tensor
    #             # keep shape as tensor and get k
    #             num_anchor = _shape_as_tensor(scores)[-2].to(device)
    #             nms_pre = torch.where(nms_pre_tensor < num_anchor,
    #                                   nms_pre_tensor, num_anchor)
    #
    #             max_scores, _ = (scores * centerness[..., None]).max(-1)
    #             _, topk_inds = max_scores.topk(nms_pre)
    #             points = points[topk_inds, :]
    #             batch_inds = torch.arange(batch_size).view(
    #                 -1, 1).expand_as(topk_inds).long()
    #             bbox_pred = bbox_pred[batch_inds, topk_inds, :]
    #             scores = scores[batch_inds, topk_inds, :]
    #             centerness = centerness[batch_inds, topk_inds]
    #         bboxes = distance2bbox(points, bbox_pred, max_shape=img_shapes)
    #         mlvl_bboxes.append(bboxes)
    #         mlvl_scores.append(scores)
    #         mlvl_centerness.append(centerness)
    #
    #     batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
    #     if rescale:
    #         batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
    #             scale_factors).unsqueeze(1)
    #     batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
    #     batch_mlvl_centerness = torch.cat(mlvl_centerness, dim=1)
    #
    #     # Set max number of box to be feed into nms in deployment
    #     deploy_nms_pre = cfg.get('deploy_nms_pre', -1)
    #     if deploy_nms_pre > 0 and torch.onnx.is_in_onnx_export():
    #         batch_mlvl_scores, _ = (
    #             batch_mlvl_scores *
    #             batch_mlvl_centerness.unsqueeze(2).expand_as(batch_mlvl_scores)
    #         ).max(-1)
    #         _, topk_inds = batch_mlvl_scores.topk(deploy_nms_pre)
    #         batch_inds = torch.arange(batch_mlvl_scores.shape[0]).view(
    #             -1, 1).expand_as(topk_inds)
    #         batch_mlvl_scores = batch_mlvl_scores[batch_inds, topk_inds, :]
    #         batch_mlvl_bboxes = batch_mlvl_bboxes[batch_inds, topk_inds, :]
    #         batch_mlvl_centerness = batch_mlvl_centerness[batch_inds,
    #                                                       topk_inds]
    #
    #     # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
    #     # BG cat_id: num_class
    #     padding = batch_mlvl_scores.new_zeros(batch_size,
    #                                           batch_mlvl_scores.shape[1], 1)
    #     batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)
    #
    #     if with_nms:
    #         det_results = []
    #         for (mlvl_bboxes, mlvl_scores,
    #              mlvl_centerness) in zip(batch_mlvl_bboxes, batch_mlvl_scores,
    #                                      batch_mlvl_centerness):
    #             det_bbox, det_label = multiclass_nms(
    #                 mlvl_bboxes,
    #                 mlvl_scores,
    #                 cfg.score_thr,
    #                 cfg.nms,
    #                 cfg.max_per_img,
    #                 score_factors=mlvl_centerness)
    #             det_results.append(tuple([det_bbox, det_label]))
    #     else:
    #         det_results = [
    #             tuple(mlvl_bs)
    #             for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores,
    #                                batch_mlvl_centerness)
    #         ]
    #     return det_results

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map sizes."""
        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        # lpf revise interface pitput: add pos_gt_ind
        labels_list, bbox_targets_list,gt_ind_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        ### lpf create flatten_gt_inds: avoid repeated gt_inds
        # lpf add 20240429: deal with repeated flatten_gt_inds across different images in a batch_size
        num_gts_list = np.array(list(map(len, gt_labels_list)))
        batch_size=len(gt_ind_list)
        cum_num_gts = list(np.cumsum(num_gts_list))  # calculate sum : right diresction one by one
        # print('get_targets : num_gts1111', num_gts_list,'batch_size',batch_size,'cum_num_gts',cum_num_gts)
        # print('num_gt_list',num_gts_list)
        # print('cum_num_gts',cum_num_gts)
        for j in range(1, batch_size):
            # print('size1',cum_num_gts[j-1],gt_inds[j].shape)
            gt_ind_list[j] += int(cum_num_gts[j - 1])


        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        # lpf add
        ## num_points : [num_points_level1,num_points_level2,num_points_level3.num_points_level4.num_points_level5]
        gt_ind_list=[gt_ind.split(num_points, 0) for gt_ind in gt_ind_list]
        # print('get_targets,shape222',batch_size,len(gt_ind_list[0]),len(gt_ind_list[1]),gt_ind_list[0][0].shape,gt_ind_list[1][1].shape,gt_ind_list[0][2].shape,gt_ind_list[1][3].shape,gt_ind_list[0][4].shape)

        # gt_inds = torch.cat(gt_ind_list)  # num_positives -- pos_inds
        # print('size2',flatten_gt_inds.shape)
        # print('cum_num_gts',cum_num_gts)
        # print('gt_inds',len(gt_inds),len(gt_inds[0]))

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        #lpf add
        concat_lvl_gt_ind = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            concat_lvl_gt_ind.append(torch.cat([gt_inds[i] for gt_inds in gt_ind_list]))

            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)

        # lpf add : concate gt_inds along the level dimension:
        # for i in range(batch_size):
        #     concat_img_gt_ind.append(torch.cat(gt_ind_list[i]))
        # print('concat_img_gt_ind.shape',concat_img_gt_ind[0].shape,concat_img_gt_ind[1].shape)
        # print('get_targets,shape33333',concat_img_gt_ind[0].shape,concat_img_gt_ind[1].shape,)
        # return concat_lvl_labels, concat_lvl_bbox_targets
        # lpf revise :

        return concat_lvl_labels, concat_lvl_bbox_targets,concat_lvl_gt_ind

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))
        # lpf areas: shape: (num_gt,)
        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        # lpf areas: new shape: (num_points, num_gt)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # condition1: inside a gt bbox
        center_sample=True# lpf add center_sample ablation 7
        if center_sample:
            # condition1: inside a `center bbox`
            radius = 1.5
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1) ## lpf min_area_inds --> indicates the gt_box's index for each sample point
        # lpf min_area's shape: (num_pints,),includes INF (back-samples)
        # print('lpf ',areas.shape,areas[1:4,:],gt_labels.shape, min_area_inds.shape)
        labels = gt_labels[min_area_inds]
        # print('lpflpf',min_area_inds[min_area_inds.nonzero()])
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        # lpf debug : min_area_inds
        # min_area_inds[min_area == INF] = -1 # mark negatives
        # pos_inds_lpf = ((labels >= 0) & (labels < self.num_classes)).nonzero().reshape(-1)
        # print('min_area_inds',min_area_inds[pos_inds_lpf])
        # for i,j in enumerate(torch.arange(num_gts)):
        #     match_get_target_single = (min_area_inds[pos_inds_lpf]==j)
        #     # print('min_area_inds',min_area_inds[pos_inds_lpf].min())
        #     print('match_get_target_single', len(match_get_target_single.nonzero()))

        # print('debug1',len(pos_inds_lpf))
        # print('lpf debug : min_area_inds',labels[pos_inds_lpf]==gt_labels[min_area_inds[((labels >= 0) & (labels < self.num_classes))]])

        # return labels, bbox_targets
        # lpf revise 20240419
        return labels, bbox_targets,min_area_inds
    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)
