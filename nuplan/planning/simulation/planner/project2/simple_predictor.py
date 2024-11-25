import numpy as np
from typing import List, Type, Optional, Tuple
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.common.actor_state.ego_state import DynamicCarState, EgoState
from nuplan.planning.simulation.planner.project2.abstract_predictor import AbstractPredictor
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.planning.simulation.trajectory.predicted_trajectory import PredictedTrajectory
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint


class SimplePredictor(AbstractPredictor):
    def __init__(self, ego_state: EgoState, observations: Observation, duration: float, sample_time: float) -> None:
        self._ego_state = ego_state
        self._observations = observations
        self._duration = duration
        self._sample_time = sample_time
        self._occupancy_map_radius = 40

    def predict_by_cv(self, object: TrackedObject):
        cur_x = object.center.x
        cur_y = object.center.y
        cur_vx = object.velocity.x
        cur_vy = object.velocity.y
        cur_heading = object.box.center.heading
        length = object.box.length
        width = object.box.width
        height = object.box.height
        # 从自车当前状态获取时间戳
        cur_time_point = self._ego_state.time_point 
        waypoint = Waypoint(time_point=cur_time_point, \
                                oriented_box=OrientedBox(center=object.center, \
                                        length=length, width=width, height=height), \
                                velocity=object.velocity
                                )
        
        waypoints: List[Waypoint] = [waypoint]
        pred_traj: List[PredictedTrajectory] = []
        for iter in range(int(self._duration.time_us / self._sample_time.time_us)):
            relative_time = (iter + 1) * self._sample_time.time_s # 计算相对时间
            pred_x = cur_x + cur_vx * relative_time  # 匀速推x位置
            pred_y = cur_y + cur_vy * relative_time  # 匀速推y位置
            pred_velocity = StateVector2D(cur_vx, cur_vy) # 匀速速度不变
            # 修改orientedBox的center变量
            pred_orientedbox = OrientedBox(center=StateSE2(pred_x, pred_y, cur_heading),
                                           length=length, width=width, height=height)
            waypoint = Waypoint(time_point=waypoint.time_point + self._sample_time, 
                                oriented_box=pred_orientedbox,
                                velocity=pred_velocity)
            waypoints.append(waypoint)
        
        pred_traj.append(PredictedTrajectory(probability=1.0, waypoints=waypoints))
        
        return pred_traj


    def predict(self):
        """Inherited, see superclass."""
        if isinstance(self._observations, DetectionsTracks):
            objects_init = self._observations.tracked_objects.tracked_objects
            objects = [
                object
                for object in objects_init
                if np.linalg.norm(self._ego_state.center.array - object.center.array) < self._occupancy_map_radius
            ]

            # TODO：1.Predicted the Trajectory of object
            for object in objects:
                if(object._tracked_object_type is TrackedObjectType.VEHICLE):
                    predicted_trajectories = self.predict_by_cv(object)  # predicted_trajectories : List[PredictedTrajectory]
                    object.predictions = predicted_trajectories
            return objects

        else:
            raise ValueError(
                f"SimplePredictor only supports DetectionsTracks. Got {self._observations.detection_type()}")
