import numpy as np

BEND = [0.0, 0.3, -0.6]
STRETCH = [0.0, -0.3, 0.6]
STAY = [0.0, 0.0, 0.0]
SOFT_BEND = [0.0, 0.15, -0.3]   # half a bend, for the bow and sit back

LEAN = 0.25   # hip angle that keeps the legs vertical while the body tilts

# leg order: FL, FR, RL, RR   (each leg: hip, thigh, calf)
LEAN_LEFT = np.array([ LEAN, 0.08, -0.16,    LEAN, -0.08, 0.16,
                       LEAN, 0.08, -0.16,    LEAN, -0.08, 0.16])
LEAN_RIGHT = np.array([-LEAN, -0.08, 0.16,  -LEAN, 0.08, -0.16,
                       -LEAN, -0.08, 0.16,  -LEAN, 0.08, -0.16])
BOW = np.array(SOFT_BEND + SOFT_BEND + STAY + STAY)
SIT_BACK = np.array(STAY + STAY + SOFT_BEND + SOFT_BEND)

DANCE = [LEAN_LEFT, LEAN_RIGHT, BOW, SIT_BACK]
MOVE_TIME = 1.0