from time import time
import numpy as np
from kdtree import KDTree, nearest_point


class SimpleTree:

    def __init__(self, dim):
        self._parents_map = {}
        self._kd = KDTree(dim)

    def insert_new_node(self, point, parent=None):
        node_id = self._kd.insert(point)
        self._parents_map[node_id] = parent

        return node_id

    def get_parent(self, child_id):
        return self._parents_map[child_id]

    def get_point(self, node_id):
        return self._kd.get_node(node_id).point

    def get_nearest_node(self, point):
        return self._kd.find_nearest_point(point)

    def construct_path_to_root(self, leaf_node_id):
        path = []
        node_id = leaf_node_id
        while node_id is not None:
            path.append(self.get_point(node_id))
            node_id = self.get_parent(node_id)

        return path


class RRTConnect:

    def __init__(self, arm, step_size=0.1):
        self.arm = arm

        self.step_size = step_size
        self.max_iterations = 5000

    def sample_valid_joints(self):
        return self.arm.sample_random_configuration()

    def extend(self, tree,q_target):
        '''
        TODO: Implement extend for RRT Connect

        - Only perform self.project_to_constraint if constraint is not None
        - Use self._is_seg_valid, self._q_step_size, self._connect_dist
        returns back if it is reached and the new node id
        '''
        nearest_id, _ = tree.get_nearest_node(q_target) # find qnear
        q_near=tree.get_point(nearest_id)
        direction = q_target-q_near
        distance=np.linalg.norm(direction)
        if distance <= self.step_size:
            q_new=q_target # connect if close enough
        else:
            q_new=q_near+self.step_size*direction/distance # take a small step towards qrand

        if not self.arm.is_within_joint_limits(q_new):
            return False,None
        if self.arm.check_collision(q_new):
            return False,None

        new_node_id=tree.insert_new_node(q_new,nearest_id) # append to the tree
        reached=np.linalg.norm(q_target-q_new)<self.step_size
        return reached,new_node_id

    def connect(self, tree, q_target):
        while True:
            reached, new_node_id = self.extend(tree, q_target) # keep trying to reach
            if reached:
                return True, new_node_id
            if new_node_id is None:
                return False, None
    def plan(self, q_start, q_target):
        tree_start=SimpleTree(self.arm.num_joints)
        tree_start.insert_new_node(q_start)
        tree_goal=SimpleTree(self.arm.num_joints)
        tree_goal.insert_new_node(q_target)
        start_is_tree_start=True # keep track of tree to prevent path backwards
        start_time=time()
        for iteration in range(self.max_iterations):
            q_rand=self.arm.sample_random_configuration()
            reached, new_node_id = self.extend(tree_start,q_rand)
            if new_node_id is not None:
                q_new=tree_start.get_point(new_node_id)
                connected, connect_node_id=self.connect(tree_goal,q_new) # try to connect from the other tree
                if connected:
                    if not start_is_tree_start: # swap it back
                        tree_start, tree_goal = tree_goal, tree_start
                        new_node_id, connect_node_id = connect_node_id, new_node_id
                    print(f"âœ“ Path found in {iteration} iterations ({time() - start_time:.2f}s)")
                    path_start=tree_start.construct_path_to_root(new_node_id)
                    path_goal=tree_goal.construct_path_to_root(connect_node_id)
                    path = path_start[::-1] + path_goal # create path backwards, no bridge needed here
                    return path
            start_is_tree_start=not start_is_tree_start # swap tree
            tree_start,tree_goal=tree_goal,tree_start
            if iteration % 1000 == 0:
                print(f"Iteration {iteration} finished")
        print("path not found!")
        return []