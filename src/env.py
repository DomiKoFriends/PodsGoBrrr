import cv2 as cv
import math
import numpy as np
from random import choice, randint, random
import torch
import pickle
from itertools import product

WIDTH = 16000
HEIGHT = 9000


class Vec2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, p):
        return math.sqrt(self.distance2(p))

    def distance2(self, p):
        return (self.x - p.x) ** 2 + (self.y - p.y) ** 2

    def dot(self, p):
        return self.x * p.x + self.y * p.y

    def cross(self, p):
        return self.x * p.y - self.y * p.x

    def norm(self):
        temp = math.sqrt(self.x ** 2 + self.y ** 2)
        return Vec2D(self.x / temp, self.y / temp)

    def trunc(self):
        return Vec2D(math.trunc(self.x), math.trunc(self.y))

    def round(self):
        return Vec2D(round(self.x), round(self.y))

    def inti(self):
        return Vec2D(int(self.x), int(self.y))

    def tupi(self):
        return (self.x, self.y)

    def angle_diff(self, p):
        return math.atan2(p.y - self.y, p.x - self.x)

    def closest(self, a, b):
        da = b.y - a.y
        db = a.x - b.x

        c1 = da * a.x + db * a.y
        c2 = -db * self.x + da * self.y
        det = da * da + db * db
        cx, cy = 0, 0
        if det != 0:
            cx = (da * c1 - db * c2) / det
            cy = (da * c2 + db * c1) / det
        else:
            cx = self.x
            cy = self.y

        return Vec2D(cx, cy)

    def __sub__(self, p):
        return Vec2D(self.x - p.x, self.y - p.y)

    def __add__(self, p):
        return Vec2D(self.x + p.x, self.y + p.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __mul__(self, p):
        return Vec2D(self.x * p, self.y * p)

    def __truediv__(self, p):
        return Vec2D(self.x / p, self.y / p)

    def __floordiv__(self, p):
        return Vec2D(self.x // p, self.y // p)

    @staticmethod
    def vec2D_by_angle(angle):
        return Vec2D(math.cos(angle), math.sin(angle))


class Unit(Vec2D):
    def __init__(self, x, y):
        super().__init__(x, y)
        # * 200, because 600-400=200, and CENTER of pod must touch CP
        self.radius = 200
        self.speed = Vec2D(0, 0)

    def collision(self, other):
        dist = self.distance2(other)
        sr = (self.radius + other.radius) ** 2

        if dist < sr:
            return (self, other, 0)

        if np.all(self.speed == other.speed):
            return None

        myp = self - other
        speed_diff = self.speed - other.speed
        up = Vec2D(0, 0)
        p = up.closest(myp, myp + speed_diff)

        pdist = up.distance2(p)

        if pdist < sr:
            length = math.sqrt((speed_diff.x ** 2) + (speed_diff.y ** 2))
            backdist = math.sqrt(sr - pdist)
            p = p - (speed_diff / length) * backdist

            if myp.distance2(p) > myp.distance2(p):
                return None

            pdist = p.distance(myp)

            if pdist > length:
                return None

            return (self, other, pdist / length)

        return None


class Pod(Unit):
    def __init__(self, x, y, angle):
        super().__init__(x, y)
        self.angle = angle
        self.radius = 400
        self.points = 0
        self.freeze_time = 0
        self.shield = False
        self.next_cp = None # used in EnvSR

    def move(self, t):
        self.x += self.speed.x * t
        self.y += self.speed.y * t

    def norm(self):
        # self.x, self.y = round(self.x), round(self.y)
        self.x, self.y = math.trunc(self.x), math.trunc(self.y)
        self.speed = (self.speed * 0.85).trunc()

        self.angle = round(self.angle * 180 / np.pi)
        self.angle = self.angle / 180 * np.pi # to rad

        while self.angle < 0: self.angle += 2 * np.pi
        while self.angle >= 2 * np.pi: self.angle -= 2 * np.pi

    def apply(self, angle, thrust=None):
        if thrust is None:
            self.shield = True
            self.freeze_time = 3
        else:
            self.shield = False
            if self.freeze_time > 0:
                self.freeze_time -= 1
            else:
                self.angle += angle
                self.speed = self.speed + \
                    Vec2D.vec2D_by_angle(self.angle) * thrust

    def bounce(self, u):
        m1 = 10 if self.shield else 1
        m2 = 10 if u.shield else 1
        mcoeff = (m1 + m2) / (m1 * m2)

        nx = self.x - u.x
        ny = self.y - u.y

        nxnysquare = nx ** 2 + ny ** 2

        dvx = self.speed.x - u.speed.x
        dvy = self.speed.y - u.speed.y

        product = nx * dvx + ny * dvy
        fx = (nx * product) / (nxnysquare * mcoeff)
        fy = (ny * product) / (nxnysquare * mcoeff)

        self.speed.x -= fx / m1
        self.speed.y -= fy / m1
        u.speed.x += fx / m2
        u.speed.y += fy / m2

        impulse = math.sqrt(fx ** 2 + fy ** 2)
        if impulse < 120:
            fx = fx * 120 / impulse
            fy = fy * 120 / impulse

        self.speed.x -= fx / m1
        self.speed.y -= fy / m1
        u.speed.x += fx / m2
        u.speed.y += fy / m2

    def is_outside(self):
        x, y = self.x, self.y

        max_dist = 50000

        return x > WIDTH + max_dist or x < -max_dist \
            or y > HEIGHT + max_dist or y < -max_dist


    def __str__(self):
        return f"x={self.x} y={self.y} sx={self.speed.x} sy={self.speed.y} a={self.angle}"

class Env():
    deg18 = np.pi / 10

    angles_cnt = 3  # must be odd
    points_cnt = 2
    
    LegalActions = list(
        product(np.linspace(-deg18, deg18, angles_cnt), [0, 200]))

    cp_reward = 1
    outside_reward = -1
    timeout_reward = -1

    max_speed = 800
    max_train_distance = 10000
    max_steps = 500
    timeout = 100

    def __init__(self, device):
        self.device = device

    def _random_checkpoint(self, idx):
        def bad_checkpoint():
            dist = self.checkpoints[idx].distance(self.checkpoints[idx - 1])
            return dist < 3 * 600 or dist > self.max_train_distance

        self.checkpoints[idx] = Unit(randint(
            0, WIDTH), randint(0, HEIGHT))

        while bad_checkpoint():
            self.checkpoints[idx] = Unit(randint(
                0, WIDTH), randint(0, HEIGHT))

    def _shift_checkpoints(self):
        for idx in range(self.points_cnt - 1):
            self.checkpoints[idx] = self.checkpoints[idx + 1]

        self._random_checkpoint(self.points_cnt - 1)

    def reset(self):
        def random_angle():
            return random() * np.pi * 2

        state = Env(self.device)

        state.time = self.timeout
        state.steps = 0

        state.pod = Pod(randint(0, WIDTH),
                           randint(0, HEIGHT), random_angle())
        state.pod.speed = Vec2D.vec2D_by_angle(
            random_angle()) * random() * self.max_speed

        def bad_first_checkpoint():
            dist = state.checkpoints[0].distance(state.pod)
            return dist < 2 * 600 or dist > self.max_train_distance / 2

        state.checkpoints = [Unit(0, 0) for _ in range(self.points_cnt)]

        state.checkpoints[0] = Unit(randint(
            0, WIDTH), randint(0, HEIGHT))

        while bad_first_checkpoint():
            state.checkpoints[0] = Unit(randint(
                0, WIDTH), randint(0, HEIGHT))

        for idx in range(1, self.points_cnt):
            state._random_checkpoint(idx)

        return state

    def get_state(self):
        result = torch.zeros(5 + (self.points_cnt - 1) * 3,
                             dtype=torch.float32, device=self.device)

        max_dist = math.sqrt(WIDTH ** 2 + HEIGHT ** 2)

        vec10 = Vec2D(1, 0)
        cp1 = self.checkpoints[0]

        distance_pod_cp1 = self.pod.distance(cp1)
        angle_pod_cp1 = self.pod.angle_diff(cp1)

        # norm every angle (angle 0 is straight to cp1)
        angle = self.pod.angle - angle_pod_cp1
        angle_dot = Vec2D.vec2D_by_angle(angle).dot(vec10)
        angle_cross = Vec2D.vec2D_by_angle(angle).cross(vec10)

        angle_v = Vec2D(0, 0).angle_diff(self.pod.speed) - angle_pod_cp1
        norm_v = math.sqrt(self.pod.speed.x ** 2 + self.pod.speed.y ** 2)
        v_dot = int(norm_v * Vec2D.vec2D_by_angle(angle_v).dot(vec10))
        v_cross = int(norm_v * Vec2D.vec2D_by_angle(angle_v).cross(vec10))

        result[0] = (distance_pod_cp1 - 200.0) / max_dist
        result[1] = angle_dot
        result[2] = angle_cross
        result[3] = v_dot / 1000.0
        result[4] = v_cross / 1000.0

        for idx in range(self.points_cnt -  1):
            cp1 = self.checkpoints[idx]
            cp2 = self.checkpoints[idx + 1]

            distance_cp1_cp2 = cp1.distance(cp2)
            angle_cp1_cp2 = cp1.angle_diff(cp2) - angle_pod_cp1
            angle_cp1_cp2_dot = Vec2D.vec2D_by_angle(angle_cp1_cp2).dot(vec10)
            angle_cp1_cp2_cross = Vec2D.vec2D_by_angle(
                angle_cp1_cp2).cross(vec10)

            result[5 + idx * 3] = angle_cp1_cp2_dot
            result[6 + idx * 3] = angle_cp1_cp2_cross
            result[7 + idx * 3] = distance_cp1_cp2 / max_dist

        return result

    def step(self, action_idx):
        next_state, reward, done = pickle.loads(pickle.dumps(self)), 0, False
        next_state.steps += 1
        next_state.time -= 1

        next_state.pod.apply(*self.LegalActions[action_idx])

        t = 0
        while t < 1:
            col = next_state.pod.collision(
                next_state.checkpoints[0])
            if col:
                a, b, col_time = col

                if t + col_time > 1:
                    next_state.pod.move(1 - t)
                    break

                next_state.pod.move(col_time)
                t += col_time

                if not isinstance(b, Pod):
                    reward += self.cp_reward
                    next_state.time = self.timeout
                    next_state.pod.points += 1

                    next_state._shift_checkpoints()
            else:
                next_state.pod.move(1 - t)
                break

        next_state.pod.norm()

        if next_state.pod.is_outside():
            reward += self.outside_reward
            done = True

        if next_state.steps == self.max_steps:
            done = True

        if next_state.time == 0:
            reward += self.timeout_reward
            done = True

        return next_state, reward, done

    def render(self):
        window = np.zeros((HEIGHT // 10, WIDTH // 10, 3), dtype='uint8')

        for cp in self.checkpoints:
            pos = (int(cp.x // 10), int(cp.y // 10))
            cv.circle(window, pos, 60, (41, 38, 100), -1)

        pods = [self.pod]

        for pod in pods:
            podPosition = (pod // 10).inti()
            p_s = (podPosition + pod.speed * 0.5).inti()
            angleArrowLen = 100
            p_a = (podPosition + Vec2D.vec2D_by_angle(pod.angle)
                   * angleArrowLen).inti()

            cv.circle(window, (int(pod.x // 10),
                      int(pod.y // 10)), 40, (100, 41, 38), -1)

            cv.arrowedLine(window, podPosition.inti().tupi(), p_s.tupi(),
                           (10, 200, 180), thickness=5, tipLength=0.2)

            cv.arrowedLine(window, podPosition.inti().tupi(), p_a.tupi(),
                           (123, 10, 67), thickness=3, tipLength=0.3)

            for checkpoint_id in range(len(self.checkpoints)):
                cv.putText(window, str(checkpoint_id + self.pod.points), ((self.checkpoints[checkpoint_id] // 10) - Vec2D(10, -10)).inti().tupi(), cv.FONT_HERSHEY_SIMPLEX,
                           1, (255, 255, 255), 2, cv.LINE_AA)

        cv.imshow("Game", window)

        cv.waitKey(50)

    def open(self):
        cv.namedWindow("Game")
        cv.moveWindow("Game", 50, 50)

    def close(self):
        cv.destroyAllWindows()


class EnvWithBlocker():
    deg18 = np.pi / 10
    angles_cnt_runner = 3  # must be odd
    angles_cnt_blocker = 3  # must be odd
    points_cnt = 2

    LegalActionsRunner = list(
        product(np.linspace(-deg18, deg18, angles_cnt_runner), [0, 200])) + [('Shield', None)]

    LegalActionsBlocker = list(
        product(np.linspace(-deg18, deg18, angles_cnt_blocker), [0, 200])) + [('Shield', None)]

    cp_reward = 1 # TODO 
    outside_reward = -1
    timeout_reward = -1
    collision_reward = 0.5 #TODO

    max_speed = 800
    max_train_distance = 10000
    max_steps = 500
    timeout = 100

    def __init__(self, device):
        self.device = device

    def _random_checkpoint(self, idx):
        def bad_checkpoint():
            dist = self.checkpoints[idx].distance(
                self.checkpoints[idx - 1])
            return dist < 3 * 600 or dist > self.max_train_distance

        self.checkpoints[idx] = Unit(randint(
            0, WIDTH), randint(0, HEIGHT))

        while bad_checkpoint():
            self.checkpoints[idx] = Unit(randint(
                0, WIDTH), randint(0, HEIGHT))

    def _shift_checkpoints(self):
        for idx in range(self.points_cnt - 1):
            self.checkpoints[idx] = self.checkpoints[idx + 1]

        self._random_checkpoint(self.points_cnt - 1)

    def reset(self):
        def random_angle():
            return random() * np.pi * 2

        state = EnvWithBlocker(self.device)

        state.time = self.timeout
        state.steps = 0

        state.runner = Pod(randint(0, WIDTH),
                           randint(0, HEIGHT), random_angle())
        state.runner.speed = Vec2D.vec2D_by_angle(
            random_angle()) * random() * self.max_speed

        state.blocker = Pod(randint(0, WIDTH),
                            randint(0, HEIGHT), random_angle())

        while state.blocker.distance(state.runner) < 600:
            state.blocker = Pod(randint(0, WIDTH),
                                randint(0, HEIGHT), random_angle())

        state.blocker.speed = Vec2D.vec2D_by_angle(
            random_angle()) * random() * self.max_speed
            
        def bad_first_checkpoint():
            dist = state.checkpoints[0].distance(state.runner)
            return dist < 2 * 600 or dist > self.max_train_distance / 2

        state.checkpoints = [Unit(0, 0) for _ in range(self.points_cnt)]

        state.checkpoints[0] = Unit(randint(
            0, WIDTH), randint(0, HEIGHT))

        while bad_first_checkpoint():
            state.checkpoints[0] = Unit(randint(
                0, WIDTH), randint(0, HEIGHT))

        for idx in range(1, self.points_cnt):
            state._random_checkpoint(idx)

        return state

    def get_state_runner(self):
        result = torch.zeros(5 + (self.points_cnt - 1) * 3,
                             dtype=torch.float32, device=self.device)

        max_dist = math.sqrt(WIDTH ** 2 + HEIGHT ** 2)

        vec10 = Vec2D(1, 0)
        cp1 = self.checkpoints[0]

        distance_pod_cp1 = self.runner.distance(cp1)
        angle_pod_cp1 = self.runner.angle_diff(cp1)

        # norm every angle (angle 0 is straight to cp1)
        angle = self.runner.angle - angle_pod_cp1
        angle_dot = Vec2D.vec2D_by_angle(angle).dot(vec10)
        angle_cross = Vec2D.vec2D_by_angle(angle).cross(vec10)

        angle_v = Vec2D(0, 0).angle_diff(self.runner.speed) - angle_pod_cp1
        norm_v = math.sqrt(self.runner.speed.x ** 2 + self.runner.speed.y ** 2)
        v_dot = int(norm_v * Vec2D.vec2D_by_angle(angle_v).dot(vec10))
        v_cross = int(norm_v * Vec2D.vec2D_by_angle(angle_v).cross(vec10))

        result[0] = (distance_pod_cp1 - 200.0) / max_dist
        result[1] = angle_dot
        result[2] = angle_cross
        result[3] = v_dot / 1000.0
        result[4] = v_cross / 1000.0

        for idx in range(self.points_cnt -  1):
            cp1 = self.checkpoints[idx]
            cp2 = self.checkpoints[idx + 1]

            distance_cp1_cp2 = cp1.distance(cp2)
            angle_cp1_cp2 = cp1.angle_diff(cp2) - angle_pod_cp1
            angle_cp1_cp2_dot = Vec2D.vec2D_by_angle(angle_cp1_cp2).dot(vec10)
            angle_cp1_cp2_cross = Vec2D.vec2D_by_angle(
                angle_cp1_cp2).cross(vec10)

            result[5 + idx * 3] = angle_cp1_cp2_dot
            result[6 + idx * 3] = angle_cp1_cp2_cross
            result[7 + idx * 3] = distance_cp1_cp2 / max_dist

        return result

    def get_state_runner_against_blocker(self):
        result = torch.zeros(10 + (self.points_cnt - 1) * 3,
                             dtype=torch.float32, device=self.device)

        max_dist = math.sqrt(WIDTH ** 2 + HEIGHT ** 2)

        vec10 = Vec2D(1, 0)
        cp1 = self.checkpoints[0]

        distance_pod_cp1 = self.runner.distance(cp1)
        angle_pod_cp1 = self.runner.angle_diff(cp1)

        # norm every angle (angle 0 is straight to cp1)
        angle = self.runner.angle - angle_pod_cp1
        angle_dot = Vec2D.vec2D_by_angle(angle).dot(vec10)
        angle_cross = Vec2D.vec2D_by_angle(angle).cross(vec10)

        angle_v = Vec2D(0, 0).angle_diff(self.runner.speed) - angle_pod_cp1
        norm_v = math.sqrt(self.runner.speed.x ** 2 + self.runner.speed.y ** 2)
        v_dot = int(norm_v * Vec2D.vec2D_by_angle(angle_v).dot(vec10))
        v_cross = int(norm_v * Vec2D.vec2D_by_angle(angle_v).cross(vec10))

        result[0] = (distance_pod_cp1 - 200.0) / max_dist
        result[1] = angle_dot
        result[2] = angle_cross
        result[3] = v_dot / 1000.0
        result[4] = v_cross / 1000.0

        distance_blocker_runner = self.blocker.distance(self.runner)
        angle_blocker_runner = self.blocker.angle_diff(self.runner)

        # norm every blocker angle (angle 0 is straight to runner)
        blocker_angle = self.blocker.angle - angle_blocker_runner
        blocker_angle_dot = Vec2D.vec2D_by_angle(blocker_angle).dot(vec10)
        blocker_angle_cross = Vec2D.vec2D_by_angle(blocker_angle).cross(vec10)

        angle_v = Vec2D(0, 0).angle_diff(self.blocker.speed) - angle_blocker_runner
        norm_v = math.sqrt(self.blocker.speed.x ** 2 + self.blocker.speed.y ** 2)
        blocker_v_dot = int(norm_v * Vec2D.vec2D_by_angle(angle_v).dot(vec10))
        blocker_v_cross = int(norm_v * Vec2D.vec2D_by_angle(angle_v).cross(vec10))

        result[5] = blocker_angle_dot
        result[6] = blocker_angle_cross
        result[7] = blocker_v_dot / 1000.0
        result[8] = blocker_v_cross / 1000.0
        result[9] = distance_blocker_runner / max_dist

        for idx in range(self.points_cnt -  1):
            cp1 = self.checkpoints[idx]
            cp2 = self.checkpoints[idx + 1]

            distance_cp1_cp2 = cp1.distance(cp2)
            angle_cp1_cp2 = cp1.angle_diff(cp2) - angle_pod_cp1
            angle_cp1_cp2_dot = Vec2D.vec2D_by_angle(angle_cp1_cp2).dot(vec10)
            angle_cp1_cp2_cross = Vec2D.vec2D_by_angle(
                angle_cp1_cp2).cross(vec10)

            result[10 + idx * 3] = angle_cp1_cp2_dot
            result[11 + idx * 3] = angle_cp1_cp2_cross
            result[12 + idx * 3] = distance_cp1_cp2 / max_dist

        return result

    def get_state_blocker(self):
        result = torch.zeros(10 + self.points_cnt * 3,
                             dtype=torch.float32, device=self.device)

        max_dist = math.sqrt(WIDTH ** 2 + HEIGHT ** 2)

        vec10 = Vec2D(1, 0)
        cp1 = self.checkpoints[0]

        distance_blocker_runner = self.blocker.distance(self.runner)
        angle_blocker_runner = self.blocker.angle_diff(self.runner)

        # norm every blocker angle (angle 0 is straight to runner)
        blocker_angle = self.blocker.angle - angle_blocker_runner
        blocker_angle_dot = Vec2D.vec2D_by_angle(blocker_angle).dot(vec10)
        blocker_angle_cross = Vec2D.vec2D_by_angle(blocker_angle).cross(vec10)

        angle_v = Vec2D(0, 0).angle_diff(self.blocker.speed) - angle_blocker_runner
        norm_v = math.sqrt(self.blocker.speed.x ** 2 + self.blocker.speed.y ** 2)
        blocker_v_dot = int(norm_v * Vec2D.vec2D_by_angle(angle_v).dot(vec10))
        blocker_v_cross = int(norm_v * Vec2D.vec2D_by_angle(angle_v).cross(vec10))

        angle_runner_cp1 = self.runner.angle_diff(cp1)
        
        # norm every angle (angle 0 is straight to cp1)
        runner_angle = self.blocker.angle - angle_runner_cp1
        runner_angle_dot = Vec2D.vec2D_by_angle(runner_angle).dot(vec10)
        runner_angle_cross = Vec2D.vec2D_by_angle(runner_angle).cross(vec10)

        angle_v = Vec2D(0, 0).angle_diff(self.runner.speed) - angle_runner_cp1
        norm_v = math.sqrt(self.runner.speed.x ** 2 + self.runner.speed.y ** 2)
        runner_v_dot = int(norm_v * Vec2D.vec2D_by_angle(angle_v).dot(vec10))
        runner_v_cross = int(norm_v * Vec2D.vec2D_by_angle(angle_v).cross(vec10))

        distance_runner_cp1 = self.runner.distance(cp1)

        result[0] = blocker_angle_dot
        result[1] = blocker_angle_cross
        result[2] = blocker_v_dot / 1000.0
        result[3] = blocker_v_cross / 1000.0
        result[4] = distance_blocker_runner / max_dist

        result[5] = runner_angle_dot
        result[6] = runner_angle_cross
        result[7] = runner_v_dot / 1000.0
        result[8] = runner_v_cross / 1000.0
        result[9] = distance_runner_cp1 / max_dist
        
        # for idx in range(self.points_cnt - 1):
        #     cp1 = self.checkpoints[idx]
        #     cp2 = self.checkpoints[idx + 1]

        #     distance_cp1_cp2 = cp1.distance(cp2)
        #     angle_cp1_cp2 = cp1.angle_diff(cp2) - angle_blocker_runner
        #     angle_cp1_cp2_dot = Vec2D.vec2D_by_angle(angle_cp1_cp2).dot(vec10)
        #     angle_cp1_cp2_cross = Vec2D.vec2D_by_angle(
        #         angle_cp1_cp2).cross(vec10)

        #     result[10 + idx * 3] = angle_cp1_cp2_dot
        #     result[11 + idx * 3] = angle_cp1_cp2_cross
        #     result[12 + idx * 3] = distance_cp1_cp2 / max_dist

        for idx in range(self.points_cnt):
            cp = self.checkpoints[idx]

            distance_blocker_cp = self.blocker.distance(cp)
            angle_blocker_cp = self.blocker.angle_diff(cp) - angle_blocker_runner
            angle_blocker_cp_dot = Vec2D.vec2D_by_angle(angle_blocker_cp).dot(vec10)
            angle_blocker_cp_cross = Vec2D.vec2D_by_angle(angle_blocker_cp).cross(vec10)

            result[10 + idx] = distance_blocker_cp / max_dist
            result[11 + idx] = angle_blocker_cp_dot
            result[12 + idx] = angle_blocker_cp_cross

        return result

    def step(self, runner_action_idx, blocker_action_idx):
        next_state, done = pickle.loads(pickle.dumps(self)), False
        reward_runner, reward_blocker = 0, 0
        next_state.steps += 1
        next_state.time -= 1
        collisions_cnt = 0

        next_state.runner.apply(*self.LegalActionsRunner[runner_action_idx])
        next_state.blocker.apply(*self.LegalActionsBlocker[blocker_action_idx])
        
        t = 0
        have_pod_collision = False
        while t < 1:
            first_col = [None]

            def upd(col):
                if col:
                    if t + col[2] > 1:
                        return

                    if first_col[0] is None or col[2] < first_col[0][2]:
                        first_col[0] = col

            upd(next_state.runner.collision(
                next_state.checkpoints[0]))

            if not have_pod_collision:
                upd(next_state.blocker.collision(next_state.runner))

            # print('after', first_col[0])

            if first_col[0]:
                a, b, col_time = first_col[0]
                t += col_time

                next_state.runner.move(col_time)
                next_state.blocker.move(col_time)

                if not isinstance(b, Pod):
                    reward_runner += self.cp_reward
                    reward_blocker -= self.cp_reward
                    next_state.time = self.timeout
                    next_state.runner.points += 1

                    next_state._shift_checkpoints()
                else:
                    have_pod_collision = True
                    collisions_cnt += 1
                    reward_blocker += self.collision_reward
                    a.bounce(b)
                    # print("after")
                    # print(f"vx = {a.speed.x} vy = {a.speed.y}")
                    # print(f"vx1 = {b.speed.x} vy1 = {b.speed.y}")
            else:
                next_state.runner.move(1 - t)
                next_state.blocker.move(1 - t)
                break

        next_state.runner.norm()
        next_state.blocker.norm()

        if next_state.runner.is_outside():
            reward_runner += self.outside_reward
            done = True

        if next_state.blocker.is_outside():
            reward_blocker += self.outside_reward

        if next_state.steps == self.max_steps:
            done = True

        if next_state.time == 0:
            reward_runner += self.timeout_reward
            reward_blocker -= self.timeout_reward
            done = True

        return next_state, reward_runner, reward_blocker, done, collisions_cnt

    def render(self):
        window = np.zeros((HEIGHT // 10, WIDTH // 10, 3), dtype='uint8')

        for cp in self.checkpoints:
            pos = (int(cp.x // 10), int(cp.y // 10))
            cv.circle(window, pos, 60, (41, 38, 100), -1)

        pods = [self.runner, self.blocker]

        for idx, pod in enumerate(pods):
            podPosition = (pod // 10).inti()
            p_s = (podPosition + pod.speed * 0.5).inti()
            angleArrowLen = 100
            p_a = (podPosition + Vec2D.vec2D_by_angle(pod.angle)
                   * angleArrowLen).inti()

            if idx == 0:
                cv.circle(window, (int(pod.x // 10),
                                   int(pod.y // 10)), 40, (100, 41, 38), -1)
            else:
                cv.circle(window, (int(pod.x // 10),
                                   int(pod.y // 10)), 40, (50, 205, 50), -1)
            if pod.freeze_time:
                cv.circle(window, (int(pod.x // 10),
                                   int(pod.y // 10)), 10 * pod.freeze_time, (255, 0, 0), -1)

            cv.arrowedLine(window, podPosition.inti().tupi(), p_s.tupi(),
                           (10, 200, 180), thickness=5, tipLength=0.2)

            cv.arrowedLine(window, podPosition.inti().tupi(), p_a.tupi(),
                           (123, 10, 67), thickness=3, tipLength=0.3)

        for checkpoint_id in range(len(self.checkpoints)):
            cv.putText(window, str(checkpoint_id + self.runner.points), ((self.checkpoints[checkpoint_id] // 10) - Vec2D(10, -10)).inti().tupi(), cv.FONT_HERSHEY_SIMPLEX,
                       1, (255, 255, 255), 2, cv.LINE_AA)

        cv.imshow("Game", window)

        cv.waitKey(50)

    def open(self):
        cv.namedWindow("Game")
        cv.moveWindow("Game", 50, 50)

    def close(self):
        cv.destroyAllWindows()


class EnvSR():
    LegalMaps = [[Unit(10353,1986),Unit(2757,4659),Unit(3358,2838)],[Unit(8179,7909),Unit(11727,5704),Unit(11009,3026),Unit(10111,1169),Unit(5835,7503),Unit(1380,2538),Unit(4716,1269),Unit(4025,5146)],[Unit(6910,1656),Unit(14908,1849),Unit(2485,3249),Unit(5533,6258),Unit(12561,1063),Unit(1589,6883),Unit(13542,2666),Unit(13967,6917)],[Unit(4912,4817),Unit(9882,5377),Unit(3692,3080),Unit(3562,1207),Unit(4231,7534),Unit(14823,6471),Unit(10974,1853),Unit(9374,3740)],[Unit(13332,4114),Unit(5874,7746),Unit(7491,4801),Unit(14268,6672),Unit(2796,1751),Unit(1039,2272),Unit(6600,1874),Unit(13467,2208)],[Unit(5674,4795),Unit(9623,7597),Unit(12512,6231),Unit(4927,3377),Unit(8358,6630),Unit(4459,7216),Unit(10301,2326),Unit(2145,3943)],[Unit(12271,7160),Unit(14203,4266),Unit(3186,5112),Unit(8012,5958),Unit(2554,6642),Unit(5870,4648),Unit(11089,2403),Unit(9144,2389)],[Unit(14086,1366),Unit(1779,2501),Unit(5391,2200),Unit(13348,4290),Unit(6144,4176),Unit(11687,5637),Unit(14990,3490),Unit(3569,7566)],[Unit(9302,5289),Unit(6419,7692),Unit(2099,4297),Unit(13329,3186),Unit(13870,7169),Unit(13469,1115),Unit(5176,5061),Unit(1260,7235)],[Unit(6752,5734),Unit(10177,7892),Unit(5146,7584),Unit(11531,1216),Unit(1596,5797),Unit(8306,3554),Unit(5814,2529),Unit(9471,5505)],[Unit(9476,3253),Unit(10312,1696),Unit(2902,6897),Unit(5072,7852),Unit(5918,1004),Unit(3176,2282),Unit(14227,2261),Unit(9986,5567)],[Unit(11141,4590),Unit(3431,6328),Unit(4284,2801)],[Unit(13048,3493),Unit(9614,3949),Unit(6999,2367),Unit(12067,4880),Unit(8525,5705),Unit(6759,5582),Unit(14646,5876),Unit(4158,3179)],[Unit(8565,6690),Unit(10713,3220),Unit(7321,7928),Unit(1578,3893),Unit(6882,2145),Unit(8878,3844),Unit(1025,7671),Unit(3637,6578)],[Unit(7483,7350),Unit(14154,4505),Unit(3917,7630),Unit(9957,6899),Unit(8070,3272),Unit(1884,1763),Unit(3155,3640),Unit(10140,1152)],[Unit(4295,7416),Unit(12579,6780),Unit(6585,5187),Unit(2804,6546),Unit(5038,4810),Unit(1702,1007),Unit(10114,1658),Unit(8425,6507)],[Unit(8333,6039),Unit(13617,5740),Unit(13457,3465),Unit(6659,7011),Unit(12132,6914),Unit(10277,1624),Unit(3740,2896),Unit(9054,7429)],[Unit(13853,1419),Unit(12855,3432),Unit(2453,7829),Unit(8173,7778),Unit(1428,4878),Unit(10194,3223),Unit(2814,2394),Unit(11452,1809)],[Unit(3144,4250),Unit(6352,4948),Unit(14725,5968),Unit(4864,7961),Unit(8442,1307),Unit(14501,3206),Unit(12630,7105),Unit(1767,6800)],[Unit(7355,6865),Unit(11967,7228),Unit(4501,7146),Unit(2977,5349),Unit(9592,4217),Unit(11713,4176),Unit(10485,2979),Unit(6139,1981)],[Unit(2589,1765),Unit(7918,4590),Unit(5921,4279),Unit(10590,2077),Unit(9780,6425),Unit(5945,6701),Unit(14440,3369),Unit(4988,2966)],[Unit(6732,4875),Unit(2541,1997),Unit(13969,3703),Unit(11421,5223),Unit(7687,7371),Unit(2560,4311),Unit(3857,5771),Unit(14273,1692)],[Unit(5125,6049),Unit(6292,4792),Unit(7679,7898),Unit(3140,5406),Unit(4676,4325),Unit(8348,3287),Unit(10258,2927),Unit(1620,1867)],[Unit(3416,2572),Unit(3994,6091),Unit(6110,7235),Unit(1493,4089),Unit(1537,7029),Unit(1594,2079),Unit(8993,5700),Unit(13129,7028)],[Unit(1613,4944),Unit(13640,7073),Unit(2072,1872),Unit(14854,3078),Unit(4484,2083),Unit(10084,5389),Unit(7002,1561),Unit(8127,6064)],[Unit(13389,4971),Unit(5988,4410),Unit(8092,1250),Unit(14259,3041),Unit(13657,6973),Unit(7445,6601),Unit(4240,5278),Unit(1662,1335)],[Unit(4290,7419),Unit(9451,1669),Unit(12371,7190),Unit(3974,3907),Unit(11155,4435),Unit(3274,2000),Unit(14666,4335),Unit(6054,1285)],[Unit(3486,5885),Unit(10938,1106),Unit(6113,2576),Unit(10667,4140),Unit(13926,1263),Unit(1638,2764),Unit(7838,7775),Unit(14491,2824)],[Unit(10449,5356),Unit(12806,1444),Unit(2108,2201),Unit(6362,3893),Unit(10672,3222),Unit(12535,6285),Unit(6657,2160),Unit(13184,4759)],[Unit(3899,5119),Unit(11225,5109),Unit(14442,2223),Unit(6176,4313),Unit(7409,5908),Unit(10162,1057),Unit(8179,7625),Unit(3765,1798)],[Unit(9663,4767),Unit(9850,7663),Unit(8480,3611),Unit(9745,2178),Unit(7110,1185),Unit(11651,3625),Unit(2446,7330),Unit(12226,7367)],[Unit(14377,7927),Unit(10282,6710),Unit(4391,4202),Unit(6951,1538),Unit(12324,6293),Unit(13854,5681),Unit(1738,6745),Unit(8578,4429)],[Unit(1515,4019),Unit(2544,5921),Unit(4463,1963),Unit(6827,5569),Unit(11309,6057),Unit(14948,4812),Unit(13964,2231),Unit(7365,3517)],[Unit(4045,6842),Unit(1465,1643),Unit(6862,6261),Unit(1491,5253),Unit(14727,3179),Unit(6082,3183),Unit(13219,4669),Unit(13310,7720)],[Unit(8957,1732),Unit(1578,6406),Unit(11732,5976),Unit(11436,2425),Unit(14054,2637),Unit(2551,3170),Unit(9647,5917),Unit(5920,3971)],[Unit(12894,4304),Unit(10314,6249),Unit(4368,6750),Unit(2885,5776),Unit(9302,4546),Unit(3233,2294),Unit(14572,4689),Unit(7955,5693)],[Unit(7237,7984),Unit(4874,2369),Unit(5346,5225),Unit(6399,3637),Unit(3165,1448),Unit(9308,1323),Unit(12931,4556),Unit(2178,5950)],[Unit(3610,6881),Unit(14942,4738),Unit(10731,1634),Unit(3880,4129),Unit(3262,1431),Unit(7435,6793),Unit(8388,4702),Unit(13784,1301)],[Unit(14682,5560),Unit(12208,5614),Unit(10579,1951),Unit(9412,3808),Unit(11739,7467),Unit(2559,5223),Unit(6841,4150),Unit(6838,6961)],[Unit(10542,1448),Unit(13016,2024),Unit(12094,4489),Unit(6045,6111),Unit(3079,6468),Unit(9520,4308),Unit(6688,1488),Unit(12845,6967)],[Unit(9212,2566),Unit(8596,4908),Unit(10021,1067),Unit(6747,5088),Unit(2067,4245),Unit(10389,6367),Unit(4507,4352),Unit(5075,1492)],[Unit(8150,7321),Unit(4285,3513),Unit(12095,3060),Unit(2383,4893),Unit(9220,1508),Unit(10207,6519),Unit(5204,7579),Unit(13766,1956)],[Unit(3627,5131),Unit(2915,7241),Unit(12639,1171),Unit(6549,3529),Unit(13500,3687),Unit(1746,5642),Unit(8351,4522),Unit(1839,3045)],[Unit(1711,3942),Unit(10892,5399),Unit(4058,1092),Unit(6112,2872),Unit(1961,6027),Unit(7148,4594),Unit(7994,1062)],[Unit(1000,1000),Unit(12000,1000),Unit(12500,2500),Unit(13000,4000),Unit(12500,5500),Unit(12000,7000)],[Unit(12000,1000),Unit(12500,2500),Unit(12500,5500),Unit(12000,7000),Unit(8000,7000),Unit(7500,5500),Unit(7500,2500),Unit(8000,1000)],[Unit(1000,4500),Unit(2500,3905),Unit(4000,5095),Unit(5500,3905),Unit(7000,5095),Unit(8500,3905),Unit(10000,5095),Unit(11500,3905)],[Unit(1000,1000),Unit(15000,8000),Unit(1000,8000),Unit(15000,1000),Unit(1000,4500),Unit(15000,4500)],[Unit(12603,1090),Unit(1043,1446),Unit(10158,1241),Unit(13789,7502),Unit(7456,3627),Unit(6218,1993),Unit(7117,6546),Unit(5163,7350)],[Unit(9214,6145),Unit(1271,7171),Unit(14407,3329),Unit(10949,2136),Unit(2443,4165),Unit(5665,6432),Unit(3079,1942),Unit(4019,5141)]]
    deg18 = np.pi / 10
    angles_cnt = 3  # must be odd
    LegalActions = list(
        product(np.linspace(-deg18, deg18, angles_cnt), [0, 200]))

    cp_reward = 1
    outside_reward = -1
    timeout_reward = -1
    points_cnt = 4

    laps_count = 3
    max_speed = 800
    max_steps = 600
    timeout = 100

    def __init__(self, device):
        self.device = device

    def reset(self, idx=None):
        state = EnvSR(self.device)

        state.time = self.timeout
        state.steps = 0

        if idx is None:
            state.checkpoints = choice(self.LegalMaps)
        else:
            state.checkpoints = self.LegalMaps[idx]

        state.checkpoint_count = len(state.checkpoints)

        cp0 = state.checkpoints[0]
        cp1 = state.checkpoints[1]

        state.pod = Pod(cp0.x, cp0.y, cp0.angle_diff(cp1))
        state.pod.next_cp = 1
        state.pod.points = 1
        state.laps = 0

        return state

    def get_state(self):
        result = torch.zeros(5 + (self.points_cnt - 1) * 4,
                             dtype=torch.float32, device=self.device)

        max_dist = math.sqrt(WIDTH ** 2 + HEIGHT ** 2)

        vec10 = Vec2D(1, 0)
        cp1 = self.checkpoints[self.pod.next_cp]

        distance_pod_cp1 = self.pod.distance(cp1)
        angle_pod_cp1 = self.pod.angle_diff(cp1)

        # norm every angle (angle 0 is straight to cp1)
        angle = self.pod.angle - angle_pod_cp1
        angle_dot = Vec2D.vec2D_by_angle(angle).dot(vec10)
        angle_cross = Vec2D.vec2D_by_angle(angle).cross(vec10)

        angle_v = Vec2D(0, 0).angle_diff(self.pod.speed) - angle_pod_cp1
        norm_v = math.sqrt(self.pod.speed.x ** 2 + self.pod.speed.y ** 2)
        v_dot = int(norm_v * Vec2D.vec2D_by_angle(angle_v).dot(vec10))
        v_cross = int(norm_v * Vec2D.vec2D_by_angle(angle_v).cross(vec10))

        result[0] = (distance_pod_cp1 - 200.0) / max_dist
        result[1] = angle_dot
        result[2] = angle_cross
        result[3] = v_dot / 1000.0
        result[4] = v_cross / 1000.0

        for idx in range(self.points_cnt -  1):
            cp1 = self.checkpoints[(self.pod.next_cp + idx) % self.checkpoint_count]
            cp2 = self.checkpoints[(self.pod.next_cp + idx + 1) % self.checkpoint_count]

            distance_cp1_cp2 = cp1.distance(cp2)
            angle_cp1_cp2 = cp1.angle_diff(cp2) - angle_pod_cp1
            angle_cp1_cp2_dot = Vec2D.vec2D_by_angle(angle_cp1_cp2).dot(vec10)
            angle_cp1_cp2_cross = Vec2D.vec2D_by_angle(
                angle_cp1_cp2).cross(vec10)
            distance_pod_cp = self.pod.distance(cp2)

            result[5 + idx * 4] = angle_cp1_cp2_dot
            result[6 + idx * 4] = angle_cp1_cp2_cross
            result[7 + idx * 4] = distance_cp1_cp2 / max_dist
            result[8 + idx * 4] = distance_pod_cp / max_dist

        return result

    def get_state_str(self):
        res = []
        res.append("dist")
        res.append("adot")
        res.append("across")
        res.append("vdot")
        res.append("vcross")

        for idx in range(self.points_cnt -  1):
            res.append("adotC")
            res.append("acrossC")
            res.append("distC")
            res.append("distP")

        return res

    def step(self, action_idx):
        next_state, reward, done = pickle.loads(pickle.dumps(self)), 0, False
        next_state.steps += 1
        next_state.time -= 1

        next_state.pod.apply(*self.LegalActions[action_idx])

        t = 0
        while t < 1:
            col = next_state.pod.collision(
                next_state.checkpoints[next_state.pod.next_cp])
            if col:
                a, b, col_time = col

                if t + col_time > 1:
                    next_state.pod.move(1 - t)
                    break

                next_state.pod.move(col_time)
                t += col_time

                if not isinstance(b, Pod):
                    reward += self.cp_reward
                    next_state.time = self.timeout
                    next_state.pod.points += 1
                    next_state.pod.next_cp = (
                        next_state.pod.next_cp + 1) % next_state.checkpoint_count
                    reward += 1

                    if next_state.pod.next_cp == 0:
                        next_state.laps += 1

                        if next_state.laps == self.laps_count:
                            done = True
            else:
                next_state.pod.move(1 - t)
                break

        next_state.pod.norm()

        if next_state.pod.is_outside():
            reward += self.outside_reward
            done = True

        if next_state.steps == self.max_steps:
            done = True

        if next_state.time == 0:
            reward += self.timeout_reward
            done = True

        return next_state, reward, done

    def render(self):
        window = np.zeros((HEIGHT // 10, WIDTH // 10, 3), dtype='uint8')

        for cp in self.checkpoints:
            pos = (int(cp.x // 10), int(cp.y // 10))
            cv.circle(window, pos, 60, (41, 38, 100), -1)

        pods = [self.pod]

        for pod in pods:
            podPosition = (pod // 10).inti()
            p_s = (podPosition + pod.speed * 0.5).inti()
            angleArrowLen = 100
            p_a = (podPosition + Vec2D.vec2D_by_angle(pod.angle)
                   * angleArrowLen).inti()

            cv.circle(window, (int(pod.x // 10),
                      int(pod.y // 10)), 40, (100, 41, 38), -1)

            cv.arrowedLine(window, podPosition.inti().tupi(), p_s.tupi(),
                           (10, 200, 180), thickness=5, tipLength=0.2)

            cv.arrowedLine(window, podPosition.inti().tupi(), p_a.tupi(),
                           (123, 10, 67), thickness=3, tipLength=0.3)

            for checkpoint_id in range(len(self.checkpoints)):
                cv.putText(window, str(checkpoint_id + self.pod.points), ((self.checkpoints[checkpoint_id] // 10) - Vec2D(10, -10)).inti().tupi(), cv.FONT_HERSHEY_SIMPLEX,
                           1, (255, 255, 255), 2, cv.LINE_AA)

        state = self.get_state().detach().cpu().numpy().tolist()
        state = " | ".join(["{:7.4f}".format(x) for x in state])
        state_str = " | ".join(["{:8}".format(x) for x in self.get_state_str()])

        cv.putText(window, state_str, (0, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
        cv.putText(window, state, (0, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

        cv.imshow("Game", window)

        cv.waitKey(50)

    def open(self):
        cv.namedWindow("Game")
        cv.moveWindow("Game", 50, 50)

    def close(self):
        cv.destroyAllWindows()


if __name__ == '__main__':
    env, done = Env('cpu').reset(), False
    print(env.LegalActions)
    env.open()
    for i in range(10):
        print("\n**********\nNEW TURN", i)
        print(env.pod)
    # while not done:
        env, reward_runner, done = env.step(3)
        env.render()
        print('rewards', reward_runner)
    env.close()