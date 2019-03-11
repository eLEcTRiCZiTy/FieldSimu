#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2019 Jan Vojnar
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np

import vpython as vp


def angle_normalize(angle):
    new_angle = angle % (np.pi * 2)
    while new_angle >= np.pi:
        new_angle -= np.pi * 2
    while new_angle < -np.pi:
        new_angle += np.pi * 2
    return new_angle


def angle_difference(first, second):
    difference = second - first
    return angle_normalize(difference)


def vector_create(x, y):
    return np.array([x, y])


def vector_unit(p):
    n = np.linalg.norm(p)
    if n > 0:
        return p / n
    else:
        return p


def vector_unit_create(x, y):
    if x == 0 and y == 0:
        return vector_create(x, y)
    else:
        return vector_unit(vector_create(x, y))


def vector_angle(a, b):
    ang1 = np.arctan2(*a[::-1])
    ang2 = np.arctan2(*b[::-1])
    return (ang1 - ang2) % (2 * np.pi)


def vector_dist(a, b):
    return np.linalg.norm(a-b)


class FieldPoint(object):
    coords = []
    radius = 0.0
    force = 0.0
    force_radius = 0.0
    actor = []
    vector = []

    def __init__(self, coords=(0.0, 0.0), radius=0.0, force=0.0, force_radius=0.0, vector=vector_create(0, 0), actor=vector_create(0, 0)):
        self.coords = np.array(coords)
        self.wall_radius = radius
        self.force = force
        self.force_radius = force_radius
        self.vector = vector
        self.actor = actor


class FieldFunctionBase(object):
    def __init__(self):
        pass

    def eval(self, point, act_coords) -> np.ndarray:
        act_force = point.force
        act_vector = (act_coords - point.coords) * act_force
        return act_vector


class FieldFunctionRepulsive(FieldFunctionBase):
    def eval(self, point, act_coords) -> np.ndarray:
        act_vector = vector_create(0.0, 0.0)
        obstacle_dist = vector_dist(point.coords, act_coords)
        if obstacle_dist == 0:
            act_vector = vector_create(float('inf'), float('inf'))
        elif obstacle_dist < point.force_radius:
            one_per_dist_to_obstacle = (1 / obstacle_dist)
            one_per_force_radius = 1 / point.force_radius
            act_force = point.force * (one_per_dist_to_obstacle - one_per_force_radius) * (1 / (
                    point.force_radius ** 2))
            act_vector = ((act_coords - point.coords) / obstacle_dist) * act_force
        return act_vector


class FieldFunctionAttractive(FieldFunctionBase):
    def eval(self, point, act_coords, actor=FieldPoint()) -> np.ndarray:
        act_vector = vector_create(0.0, 0.0)
        goal_dist = vector_dist(point.coords, actor.coords)
        if goal_dist == 0:
            act_vector = vector_create(float('inf'), float('inf'))
        else:
            act_force = - point.force * np.sqrt(goal_dist)
            act_vector = vector_unit(actor.coords - point.coords) * act_force
        return act_vector


class FieldFunctionRepulsiveVectored(FieldFunctionBase):
    def eval(self, point, act_coords) -> np.ndarray:
        act_vector = vector_create(0.0, 0.0)
        obstacle_dist = vector_dist(point.coords, act_coords)
        if obstacle_dist == 0:
            act_vector = vector_create(float('inf'), float('inf'))
        elif obstacle_dist < point.force_radius:
            one_per_dist_to_obstacle = (1 / obstacle_dist)
            one_per_force_radius = 1 / point.force_radius
            act_force = point.force * (one_per_dist_to_obstacle - one_per_force_radius) * (1 / (
                    point.force_radius ** 2))
            act_vector = (point.vector / obstacle_dist) * act_force
        return act_vector


class FieldFunctionFlowVectored(FieldFunctionBase):
    def eval(self, point, act_coords) -> np.ndarray:
        act_vector = vector_unit(point.vector) * point.force
        return act_vector




class Field(object):
    repulsive_points = []
    attractive_points = []
    flow_point = FieldPoint()
    actor_point = FieldPoint()

    repulsive_model = FieldFunctionBase()
    attraction_model = FieldFunctionBase()
    flow_model = FieldFunctionBase()

    def __init__(self, repulsive_model=FieldFunctionRepulsive(), attractive_model=FieldFunctionAttractive(),
                 flow_model=FieldFunctionFlowVectored()):
        self.repulsive_model = repulsive_model
        self.attraction_model = attractive_model
        self.flow_model = flow_model
        pass

    def add_repulsive_point(self, field_point: FieldPoint):
        self.repulsive_points.append(field_point)

    def add_attractive_point(self, field_point: FieldPoint):
        self.attractive_points.append(field_point)

    def set_flow(self, field_point: FieldPoint):
        self.flow_point = field_point

    def set_actor_position(self, field_point: FieldPoint):
        self.actor_point = field_point

    def get_field_point(self, coords: np.ndarray):
        act_coords = coords

        def apply_rep_eval(point):
            return self.repulsive_model.eval(point, act_coords)

        def apply_att_eval(point):
            return self.attraction_model.eval(point, act_coords, actor=self.actor_point)

        act_vector = self.flow_model.eval(self.flow_point, act_coords)  # vector_create(0.0, 0.0)
        act_vector += sum(map(apply_rep_eval, self.repulsive_points))
        act_vector += sum(map(apply_att_eval, self.attractive_points))

        return act_vector

    def get_field_array(self, width: int, height: int):
        array_x = np.zeros([width, height])
        array_y = np.zeros([width, height])
        max_force = max(*self.repulsive_points, key=lambda x: x.force)

        for coord_x in range(0, width):
            for coord_y in range(0, height):

                act_vector = self.get_field_point(vector_create(coord_x, coord_y))

                # NOTE: only for clearer visualisation but can corrupt path finding:
                norm = np.linalg.norm(act_vector)
                act_vector = act_vector if norm < 10 else np.array([0.0, 0.0])

                array_x[coord_x, coord_y] = act_vector[0]
                array_y[coord_x, coord_y] = act_vector[1]
        return array_x, array_y


def get_sensor_points(points, actor, actor_angle, distance):
    angle_points = dict()
    output = []
    for point in points:
        actor_vector = vector_unit_create(np.cos(actor_angle), np.sin(actor_angle))
        dist_vector = -actor + point.coords
        dist = np.linalg.norm(dist_vector)
        angle = angle_normalize(vector_angle(actor_vector, dist_vector))
        if np.abs(angle) < np.pi/2:
            if dist <= distance:
                angle_points[angle] = point.coords
    return angle_points


class Simulator(object):

    width = 40
    height = 40
    show_field = False

    # visual objects:
    scene = None
    visual_walls = None
    visual_actor = None
    visual_arrows = None
    visual_ftg_points = None
    visual_ftg_goal_points = None
    visual_ftg_gap_points = None

    # logic objects:
    field = None

    ftg_points = []
    ftg_gap_points = []
    ftg_goal_point = None

    car_position = vector_create(6.0, 32.0)
    car_magnitude = 0.0
    car_angle = 0.0

    car_position_next = car_position
    car_magnitude_next = car_magnitude
    car_angle_next = car_angle

    flow_force = 0.5
    wall_radius = 2.0
    wall_force = 400.0
    wall_force_radius = 10.0

    def __init__(self, scene, show_field=False):
        self.scene = scene
        self.show_field = show_field
        self.field = Field(repulsive_model=FieldFunctionRepulsive(), attractive_model=FieldFunctionAttractive())

        self.visual_arrows = [[vp.arrow(canvas=self.scene, pos=vp.vector(x, y, 0), visible=False) for x in range(self.width)] for y in range(self.height)]
        self.visual_walls = [[list() for x in range(self.width)] for y in range(self.height)]
        self.visual_ftg_points = vp.points(canvas=self.scene, radius=10, color=vp.color.green)
        self.visual_ftg_gap_points = vp.points(canvas=self.scene, radius=10, color=vp.vector(0.9,1.0,0.0))
        self.visual_ftg_goal_points = vp.points(canvas=self.scene, radius=10, color=vp.vector(1.0,0.6,0.0))

    def create_walls(self):
        wall_radius_opacity = 0.2

        height = self.height
        width = self.width

        size_x = 0.5
        size_y = 0.5
        size_z = 2

        size_x_wide = 1
        size_y_wide = 1

        for iy in np.arange(height*0, height*1, 1):
            x = 0 * width
            y = iy
            z = 0
            self.field.add_repulsive_point(FieldPoint([x, y], self.wall_radius, self.wall_force, self.wall_force_radius))
            self.visual_walls[int(x)][int(y)].append(vp.box(canvas=self.scene, pos=vp.vector(x, y, z), size=vp.vector(size_x, size_y_wide, size_z), color=vp.color.red))
            self.visual_walls[int(x)][int(y)].append(vp.sphere(canvas=self.scene, pos=vp.vector(x, y, z), radius=self.wall_radius, color=vp.color.red, opacity=wall_radius_opacity))

        for iy in np.arange(height*0.3, height*0.7, 1):
            x = 0.3 * width
            y = iy
            z = 0
            self.field.add_repulsive_point(FieldPoint([x, y], self.wall_radius, self.wall_force, self.wall_force_radius))
            self.visual_walls[int(x)][int(y)].append(vp.box(canvas=self.scene, pos=vp.vector(x, y, z), size=vp.vector(size_x, size_y_wide, size_z), color=vp.color.red))
            self.visual_walls[int(x)][int(y)].append(vp.sphere(canvas=self.scene, pos=vp.vector(x, y, z), radius=self.wall_radius, color=vp.color.red, opacity=wall_radius_opacity))

        for iy in np.arange(height*0, height*1, 1):
            x = 1 * width - 1
            y = iy
            z = 0
            self.field.add_repulsive_point(FieldPoint([x, y], self.wall_radius, self.wall_force, self.wall_force_radius))
            self.visual_walls[int(x)][int(y)].append(vp.box(canvas=self.scene, pos=vp.vector(x, y, z), size=vp.vector(size_x, size_y_wide, size_z), color=vp.color.red))
            self.visual_walls[int(x)][int(y)].append(vp.sphere(canvas=self.scene, pos=vp.vector(x, y, z), radius=self.wall_radius, color=vp.color.red, opacity=wall_radius_opacity))

        for iy in np.arange(height*0.3, height*0.7, 1):
            x = 0.7 * width
            y = iy
            z = 0
            self.field.add_repulsive_point(FieldPoint([x, y], self.wall_radius, self.wall_force, self.wall_force_radius))
            self.visual_walls[int(x)][int(y)].append(vp.box(canvas=self.scene, pos=vp.vector(x, y, z), size=vp.vector(size_x, size_y_wide, size_z), color=vp.color.red))
            self.visual_walls[int(x)][int(y)].append(vp.sphere(canvas=self.scene, pos=vp.vector(x, y, z), radius=self.wall_radius, color=vp.color.red, opacity=wall_radius_opacity))

        for iy in np.arange(height*0.7, height*0.8, 1):
            x = 0.5 * width
            y = iy
            z = 0
            self.field.add_repulsive_point(FieldPoint([x, y], self.wall_radius, self.wall_force, self.wall_force_radius))
            self.visual_walls[int(x)][int(y)].append(vp.box(canvas=self.scene, pos=vp.vector(x, y, z), size=vp.vector(size_x, size_y_wide, size_z), color=vp.color.red))
            self.visual_walls[int(x)][int(y)].append(vp.sphere(canvas=self.scene, pos=vp.vector(x, y, z), radius=self.wall_radius, color=vp.color.red, opacity=wall_radius_opacity))

        for ix in np.arange(1, width*1, 1):
            x = ix
            y = 1
            z = 0
            self.field.add_repulsive_point(FieldPoint([x, y], self.wall_radius, self.wall_force, self.wall_force_radius))
            self.visual_walls[int(x)][int(y)].append(vp.box(canvas=self.scene, pos=vp.vector(x, y, z), size=vp.vector(size_x_wide, size_y, size_z), color=vp.color.red))
            self.visual_walls[int(x)][int(y)].append(vp.sphere(canvas=self.scene, pos=vp.vector(x, y, z), radius=self.wall_radius, color=vp.color.red, opacity=wall_radius_opacity))

        for ix in np.arange(width*0.3 + 1, width*0.7 - 1, 1):
            x = ix
            y = 0.3 * height
            z = 0
            self.field.add_repulsive_point(FieldPoint([x, y], self.wall_radius, self.wall_force, self.wall_force_radius))
            self.visual_walls[int(x)][int(y)].append(vp.box(canvas=self.scene, pos=vp.vector(x, y, z), size=vp.vector(size_x_wide, size_y, size_z), color=vp.color.red))
            self.visual_walls[int(x)][int(y)].append(vp.sphere(canvas=self.scene, pos=vp.vector(x, y, z), radius=self.wall_radius, color=vp.color.red, opacity=wall_radius_opacity))

        for ix in np.arange(width*0.7 +1, width*0.8, 1):
            x = ix
            y = 0.3* height
            z = 0
            self.field.add_repulsive_point(FieldPoint([x, y], self.wall_radius, self.wall_force, self.wall_force_radius))
            self.visual_walls[int(x)][int(y)].append(vp.box(canvas=self.scene, pos=vp.vector(x, y, z), size=vp.vector(size_x_wide, size_y, size_z), color=vp.color.red))
            self.visual_walls[int(x)][int(y)].append(vp.sphere(canvas=self.scene, pos=vp.vector(x, y, z), radius=self.wall_radius, color=vp.color.red, opacity=wall_radius_opacity))

            y = 0.65 * height
            self.field.add_repulsive_point(FieldPoint([x, y], self.wall_radius, self.wall_force, self.wall_force_radius))
            self.visual_walls[int(x)][int(y)].append(vp.box(canvas=self.scene, pos=vp.vector(x, y, z), size=vp.vector(size_x_wide, size_y, size_z), color=vp.color.red))
            self.visual_walls[int(x)][int(y)].append(vp.sphere(canvas=self.scene, pos=vp.vector(x, y, z), radius=self.wall_radius, color=vp.color.red, opacity=wall_radius_opacity))

        for ix in np.arange(width*0.85, width*1 - 1, 1):
            x = ix
            y = 0.5 * height
            z = 0
            self.field.add_repulsive_point(FieldPoint([x, y], self.wall_radius, self.wall_force, self.wall_force_radius))
            self.visual_walls[int(x)][int(y)].append(vp.box(canvas=self.scene, pos=vp.vector(x, y, z), size=vp.vector(size_x_wide, size_y, size_z), color=vp.color.red))
            self.visual_walls[int(x)][int(y)].append(vp.sphere(canvas=self.scene, pos=vp.vector(x, y, z), radius=self.wall_radius, color=vp.color.red, opacity=wall_radius_opacity))

            y = 0.8 * height
            self.field.add_repulsive_point(FieldPoint([x, y], self.wall_radius, self.wall_force, self.wall_force_radius))
            self.visual_walls[int(x)][int(y)].append(vp.box(canvas=self.scene, pos=vp.vector(x, y, z), size=vp.vector(size_x_wide, size_y, size_z), color=vp.color.red))
            self.visual_walls[int(x)][int(y)].append(vp.sphere(canvas=self.scene, pos=vp.vector(x, y, z), radius=self.wall_radius, color=vp.color.red, opacity=wall_radius_opacity))

        for ix in np.arange(1, width*1 - 1, 1):
            x = ix
            y = width * 1 - 2
            z = 0
            self.field.add_repulsive_point(FieldPoint([x, y], self.wall_radius, self.wall_force, self.wall_force_radius))
            self.visual_walls[int(x)][int(y)].append(vp.box(canvas=self.scene, pos=vp.vector(x, y, z), size=vp.vector(size_x_wide, size_y, size_z), color=vp.color.red))
            self.visual_walls[int(x)][int(y)].append(vp.sphere(canvas=self.scene, pos=vp.vector(x, y, z), radius=self.wall_radius, color=vp.color.red, opacity=wall_radius_opacity))

        for ix in np.arange(width*0.3 + 1, width*0.7, 1):
            x = ix
            y = 0.7 * height
            z = 0
            self.field.add_repulsive_point(FieldPoint([x, y], self.wall_radius, self.wall_force, self.wall_force_radius))
            self.visual_walls[int(x)][int(y)].append(vp.box(canvas=self.scene,  pos=vp.vector(x, y, z), size=vp.vector(size_x_wide, size_y, size_z), color=vp.color.red))
            self.visual_walls[int(x)][int(y)].append(vp.sphere(canvas=self.scene, pos=vp.vector(x, y, z), radius=self.wall_radius, color=vp.color.red, opacity=wall_radius_opacity))

    def process_ftg(self):
        ftg_points_dict = get_sensor_points(self.field.repulsive_points, self.car_position, self.car_angle, 16)
        self.ftg_points = ftg_points_dict.values()
        last_key = 0
        last_value = []
        largest_angle = 0
        largest_pair = []
        first = True
        sorted_points_dict = sorted(ftg_points_dict.items())
        for key, value in sorted_points_dict:
            if (key > np.pi / 2) or (key < -np.pi / 2):
                continue
            if first:
                first = False
            else:
                angle = np.abs(last_key - key)
                if angle > largest_angle:
                    largest_angle = angle
                    largest_pair = [last_value, value]
            last_key = key
            last_value = value

        self.ftg_gap_points = largest_pair

        # largest_pair_np = np.array(largest_pair)
        ftg_point = None
        if largest_angle > 0.3:
            if len(largest_pair) > 1:
                ftg_point = np.mean(largest_pair, axis=0)
                self.field.attractive_points.clear()
                self.field.add_attractive_point(FieldPoint(ftg_point, force=1.5))

        self.ftg_goal_point = ftg_point

    def init_logic(self):
        self.create_walls()
        self.field.set_actor_position(FieldPoint([self.car_position[0], self.car_position[0]]))
        flow_vector = vector_create(np.cos(self.car_angle), np.sin(self.car_angle))
        flow_point = FieldPoint(coords=self.car_position, radius=1, force=self.flow_force, force_radius=200,
                                vector=flow_vector)
        self.field.set_flow(flow_point)

    def update_logic(self):
        new_delta = self.field.get_field_point(self.car_position)

        # compute angle and magnitude
        new_angle = vector_angle(new_delta, vector_unit_create(1, 0))  # vector_create(car_x, car_y))
        new_angle = angle_normalize(new_angle)
        new_magnitude = np.linalg.norm(new_delta)
        new_magnitude = max(min(new_magnitude, 1), 0.05)

        # compute new magnitude + limit it
        delta_magnitude = np.fabs(new_magnitude - self.car_magnitude)
        self.car_magnitude_next = self.car_magnitude
        if self.car_magnitude < delta_magnitude:
            self.car_magnitude_next += min(delta_magnitude, 0.1)
        else:
            self.car_magnitude_next -= min(delta_magnitude, 0.2)

        # compute new angle + limit it
        delta_angle = angle_difference(self.car_angle, new_angle)
        self.car_angle_next = self.car_angle
        if delta_angle > 0.0:
            self.car_angle_next += min(np.pi / (5 + (10 * self.car_magnitude)), delta_angle)
        if delta_angle < 0.0:
            self.car_angle_next += max(-np.pi / (5 + (10 * self.car_magnitude)), delta_angle)
        self.car_angle_next = angle_normalize(self.car_angle_next)

        # compute new position delta + limit displacement per step
        new_delta = vector_create(np.cos(self.car_angle_next), np.sin(self.car_angle_next)) * max(min(self.car_magnitude_next, 1.0), 0.01)
        self.car_position_next = self.car_position + new_delta

        # update fields
        self.field.set_actor_position(FieldPoint([self.car_position[0], self.car_position[1]]))
        flow_vector = new_delta
        flow_point = FieldPoint(coords=self.car_position, radius=1, force=self.flow_force, force_radius=20,
                                vector=flow_vector)
        self.field.set_flow(flow_point)
        self.process_ftg()

        # step
        self.car_position = self.car_position_next
        self.car_angle = self.car_angle_next
        self.car_magnitude = self.car_magnitude_next



    def init_visual(self):
        car_vector = vp.vector(np.cos(self.car_angle), np.sin(self.car_angle), 0)
        self.visual_actor = vp.arrow(canvas=self.scene, shaftwidth=1, headwidth=1.5, headlenght=3,
                              pos=vp.vector(self.car_position[0], self.car_position[1], 0), axis=car_vector, make_trail=True,
                              trail_radius=0.1, retain=50)

    def update_visual(self):
        self.visual_ftg_points.clear()
        if self.ftg_points:
            self.visual_ftg_points.append([vp.vector(p[0], p[1], 1) for p in self.ftg_points])

        self.visual_ftg_gap_points.clear()
        if self.ftg_gap_points:
            self.visual_ftg_gap_points.append([vp.vector(p[0], p[1], 1) for p in self.ftg_gap_points])

        self.visual_ftg_goal_points.clear()
        if isinstance(self.ftg_goal_point, np.ndarray) and self.ftg_goal_point.size:
            self.visual_ftg_goal_points.append([vp.vector(self.ftg_goal_point[0], self.ftg_goal_point[1], 1)])

        if self.show_field:
            def update_arrow(coords):
                coord_x, coord_y = coords
                act_vector = self.field.get_field_point(vector_create(coord_x, coord_y))
                arrow = self.visual_arrows[coord_x][coord_y]
                # NOTE: only for clearer visualisation but can corrupt path finding:
                norm = np.linalg.norm(act_vector)
                act_vector = act_vector / 10
                if norm < 20:
                    arrow.visible = True
                    arrow.axis = vp.vector(act_vector[0], act_vector[1], 0)
                else:
                    arrow.visible = False

            to_process = list(itertools.product(range(0, self.width, 4), range(0, self.height, 4)))
            list(map(update_arrow, to_process))

        vp_car_vector = vp.vector(np.cos(self.car_angle), np.sin(self.car_angle), 0)
        self.visual_actor.axis = vp_car_vector * 2
        vp_car_position = vp.vector(self.car_position[0], self.car_position[1], 0)
        self.visual_actor.pos = vp_car_position

    def start(self):
        i_cycle = 0

        self.init_logic()
        self.init_visual()
        #self.update_visual()

        while True:
            vp.rate(20)
            print(i_cycle)

            self.update_logic()
            self.update_visual()

            i_cycle += 1


if __name__ == '__main__':
    race_scene = vp.canvas(title='Race', center=vp.vector(20, 20, 0))
    sim = Simulator(race_scene)
    sim.start()
