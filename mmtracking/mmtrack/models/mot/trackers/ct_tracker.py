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
              frame_id,
              public_bboxes,
              public_labels):
        valid_inds = bboxes[:, -1] > self.obj_score_thr
        bboxes_input = bboxes_input[valid_inds]
        bboxes = bboxes[valid_inds]
        bboxes_with_motion = bboxes_with_motion[valid_inds]
        det_centers_with_motion = self._xyxy2center(bboxes_with_motion)
        labels = labels[valid_inds]
        item_size = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])  # N
        ids = torch.full((bboxes.size(0),), -1, dtype=torch.long)
        pre_bboxes = self.pre_bboxes
        N = bboxes.shape[0]
        if self.empty or bboxes.size(0) == 0 or pre_bboxes is None:
            if public_bboxes is not None:
                p_dist = torch.cdist(det_centers_with_motion, self._xyxy2center(public_bboxes))
                p_invalid = p_dist > item_size.reshape(N, 1)
                p_dist += p_invalid * 1e18
                p_matched_indices = self._greedy_assignment(p_dist)
                ids[p_matched_indices[:, 0]] = torch.arange(
                    self.num_tracks,
                    self.num_tracks + p_matched_indices.shape[0],
                    dtype=torch.long)

                matched = ids != -1

                ids = ids[matched]
                bboxes_input = bboxes_input[matched]
                bboxes = bboxes[matched]
                labels = labels[matched]

                self.num_tracks += p_matched_indices.shape[0]
            else:
                num_new_tracks = bboxes.size(0)
                ids = torch.arange(
                    self.num_tracks,
                    self.num_tracks + num_new_tracks,
                    dtype=torch.long)
                self.num_tracks += num_new_tracks
        else:
            M = pre_bboxes.shape[0]
            track_size = (pre_bboxes[:, 3] - pre_bboxes[:, 1]) * \
                         (pre_bboxes[:, 2] - pre_bboxes[:, 0])  # M
            dist = torch.cdist(det_centers_with_motion, self.pre_cts, 2)
            # invalid
            invalid = ((dist > track_size.reshape(1, M)) + \
                       (dist > item_size.reshape(N, 1)) + \
                       (labels.reshape(N, 1) != self.pre_labels.reshape(1, M))) > 0
            dist = dist + invalid * 1e18
            matched_indices = self._greedy_assignment(dist)
            # pre_ids = self.pre_ids
            ids[matched_indices[:, 0]] = self.pre_ids[matched_indices[:, 1]]

            # public detection
            if public_bboxes is not None:
                p_dist = torch.cdist(self._xyxy2center(public_bboxes),det_centers_with_motion)
                # Filter out bbox matched with previous frame
                p_dist[:,matched_indices[:, 0]] += 1e18
                p_invalid = p_dist > item_size.reshape(1,N)
                p_dist += p_invalid * 1e18
                p_matched_indices = self._greedy_assignment(p_dist)
                ids[p_matched_indices[:, 1]] = torch.arange(
                    self.num_tracks,
                    self.num_tracks + p_matched_indices.shape[0],
                    dtype=torch.long)

                matched = ids != -1

                ids = ids[matched]
                bboxes_input = bboxes_input[matched]
                bboxes = bboxes[matched]
                labels = labels[matched]
                self.num_tracks += p_matched_indices.shape[0]
            else:
                new_track_inds = ids == -1
                ids[new_track_inds] = torch.arange(
                    self.num_tracks,
                    self.num_tracks + new_track_inds.sum(),
                    dtype=torch.long)
                self.num_tracks += new_track_inds.sum()
        self.update(
            ids=ids,
            bboxes_input=bboxes_input,
            bboxes=bboxes,
            cts=self._xyxy2center(bboxes),
            labels=labels,
            frame_ids=frame_id)
        return bboxes, labels, ids

    @property
    def bboxes_input(self):
        bboxes = [track['bboxes_input'] for id, track in self.tracks.items()]
        if len(bboxes) == 0:
            return None
        return torch.cat(bboxes, 0)

    def pre_active_bboxes_input(self,frame_id):
        bboxes = []
        for id, track in self.tracks.items():
            if frame_id - track['frame_ids'] == 1:
                bboxes.append(track['bboxes_input'])
        if len(bboxes) == 0:
            return None
        return torch.cat(bboxes, 0)

    @property
    def pre_bboxes(self):
        bboxes = [track['bboxes'] for id, track in self.tracks.items()]
        if len(bboxes) == 0:
            return None
        return torch.cat(bboxes, 0)

    @property
    def pre_cts(self):
        cts = [track['cts'] for id, track in self.tracks.items()]
        if len(cts) == 0:
            return None
        return torch.cat(cts, 0)

    @property
    def pre_labels(self):
        labels = [track['labels'] for id, track in self.tracks.items()]
        if len(labels) == 0:
            return None
        return torch.cat(labels, 0)

    @property
    def pre_ids(self):
        ids = [track['ids'] for id, track in self.tracks.items()]
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