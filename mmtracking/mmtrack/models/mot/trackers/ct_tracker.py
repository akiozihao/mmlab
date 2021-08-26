import numpy as np
import torch
from mmtrack.models import TRACKERS

from .base_tracker import BaseTracker


@TRACKERS.register_module()
class CTTracker(BaseTracker):
    def __init__(self,
                 obj_score_thr=0.4,
                 momentums=None,
                 num_frames_retain=3):
        super(CTTracker, self).__init__(momentums, num_frames_retain)
        self.obj_score_thr = obj_score_thr

    def track(self,
              bboxes_input,
              bboxes,
              bboxes_with_motion,
              labels,
              frame_id):
        valid_inds = bboxes[:, -1] > self.obj_score_thr
        bboxes_input = bboxes_input[valid_inds]
        bboxes = bboxes[valid_inds]
        bboxes_with_motion = bboxes_with_motion[valid_inds]
        labels = labels[valid_inds]

        pre_bboxes = self.pre_bboxes
        active = torch.ones(bboxes.size(0), dtype=torch.long)
        if self.empty or bboxes.size(0) == 0 or pre_bboxes is None:
            num_new_tracks = bboxes.size(0)
            ids = torch.arange(
                self.num_tracks,
                self.num_tracks + num_new_tracks,
                dtype=torch.long)
            self.num_tracks += num_new_tracks
        else:
            ids = torch.full((bboxes.size(0),), -1, dtype=torch.long)
            M = pre_bboxes.shape[0]
            N = bboxes.shape[0]
            track_size = (pre_bboxes[:, 3] - pre_bboxes[:, 1]) * \
                         (pre_bboxes[:, 2] - pre_bboxes[:, 0])  # M
            item_size = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])  # N
            det_centers_with_motion = self._xyxy2center(bboxes_with_motion)
            dist = torch.cdist(det_centers_with_motion, self.pre_cts, 2)
            # invalid
            invalid = ((dist > track_size.reshape(1, M)) + \
                       (dist > item_size.reshape(N, 1)) + \
                       (labels.reshape(N, 1) != self.pre_labels.reshape(1, M))) > 0
            dist = dist + invalid * 1e18
            matched_indices = self._greedy_assignment(dist)
            ids[matched_indices[:, 0]] = self.pre_ids[matched_indices[:, 1]]
            new_track_inds = ids == -1
            ids[new_track_inds] = torch.arange(
                self.num_tracks,
                self.num_tracks + new_track_inds.sum(),
                dtype=torch.long)
            self.num_tracks += new_track_inds.sum()

            # deactivate unmatched track
            for k, v in self.tracks.items():
                if k not in self.pre_ids[matched_indices[:, 1]]:
                    v['active'][-1] &= 0

        self.update(
            ids=ids,
            bboxes_input=bboxes_input,
            bboxes=bboxes,
            cts=self._xyxy2center(bboxes),
            labels=labels,
            active=active,
            frame_ids=frame_id)
        return bboxes, labels, ids

    @property
    def bboxes_input(self):
        bboxes = [track['bboxes_input'][-1] for id, track in self.tracks.items()]
        if len(bboxes) == 0:
            return None
        return torch.cat(bboxes, 0)

    @property
    def pre_active_bboxes_input(self):
        bboxes = []
        for id, track in self.tracks.items():
            if track['active'][-1] == 1:
                bboxes.append(track['bboxes_input'][-1])
        if len(bboxes) == 0:
            return None
        return torch.cat(bboxes, 0)

    @property
    def pre_bboxes(self):
        bboxes = [track['bboxes'][-1] for id, track in self.tracks.items()]
        if len(bboxes) == 0:
            return None
        return torch.cat(bboxes, 0)

    @property
    def pre_cts(self):
        cts = [track['cts'][-1] for id, track in self.tracks.items()]
        if len(cts) == 0:
            return None
        return torch.cat(cts, 0)

    @property
    def pre_labels(self):
        labels = [track['labels'][-1] for id, track in self.tracks.items()]
        if len(labels) == 0:
            return None
        return torch.cat(labels, 0)

    @property
    def pre_ids(self):
        ids = [track['ids'][-1] for id, track in self.tracks.items()]
        if len(ids) == 0:
            return None
        return torch.cat(ids, 0)

    def _xyxy2center(self, bbox):  # shape (N,5)
        ctx = bbox[:, 0] + (bbox[:, 2] - bbox[:, 0]) / 2
        cty = bbox[:, 1] + (bbox[:, 3] - bbox[:, 1]) / 2
        return torch.cat((ctx.reshape(-1, 1), cty.reshape(-1, 1)), 1)

    def _greedy_assignment(self, dist):
        dist = dist.cpu().numpy()
        matched_indices = []
        if dist.shape[1] == 0:
            return np.array(matched_indices, np.int32).reshape(-1, 2)
        for i in range(dist.shape[0]):
            j = dist[i].argmin()
            if dist[i][j] < 1e16:
                dist[:, j] = 1e18
                matched_indices.append([i, j])
        return np.array(matched_indices, np.int32).reshape(-1, 2)
