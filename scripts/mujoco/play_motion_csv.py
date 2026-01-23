import mujoco, mujoco_viewer
from tqdm import tqdm
from robolab.assets import ISAAC_DATA_DIR
from pynput import keyboard
import os
import csv
import time

class cmd:
    reset_requested = False
    @classmethod
    def reset(cls):
        """reset all velocities to zero"""
        cls.vx = 0.0
        cls.vy = 0.0
        cls.dyaw = 0.0
        print(f"Velocities reset: vx: {cls.vx:.2f}, vy: {cls.vy:.2f}, dyaw: {cls.dyaw:.2f}")

def on_press(key):
    """Key press event handler"""
    try:
        # Number key controls: 8/5 control forward/backward (vx), 4/6 control left/right (vy), 7/9 control left/right turn (dyaw)
        if hasattr(key, 'char') and key.char is not None:
            c = key.char.lower()
            if c == '0':
                # 8 -> forward (increase vx)
                cmd.reset_requested = True
                print('Reset requested (0 key pressed)')
    except AttributeError:
        pass

def on_release(key):
    """Key release event handler"""
    # If movement should only occur while keys are held down, handle it here
    pass

def start_keyboard_listener():
    """Start keyboard listener"""
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return listener

def load_qpos_from_csv(csv_file):
    qpos_list = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            q = [float(val) for val in row]
            qpos=q.copy()
            qpos[3:7]=[q[6],q[3],q[4],q[5]]
            qpos_list.append(qpos)
    return qpos_list

def run_mujoco(cfg,loop=False,motion_file=None):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.
        headless: If True, run without GUI and save video.

    Returns:
        None
    """
    print("=" * 60)
    keyboard_listener = start_keyboard_listener()

    qpos_list=load_qpos_from_csv(motion_file)

    num_frames = len(qpos_list)
    def frame_idx(t):
        if loop and num_frames > 0:
            return t % num_frames
        return t if t < num_frames else num_frames - 1

    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    data.qpos=qpos_list[0]
    mujoco.mj_fwdPosition(model, data)

    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()

    os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
    os.environ['MUJOCO_GL'] = 'glfw'
    
    mode = 'window'
    viewer = mujoco_viewer.MujocoViewer(model, data,width=1920, height=1080)
    viewer.cam.distance = 4.0
    viewer.cam.azimuth = 45.0
    viewer.cam.elevation = -20.0
    viewer.cam.lookat = [0, 0, 1]
    viewer.render()

    count_lowlevel = 0
    start_time = time.time()
    for step in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
        if cmd.reset_requested:
            print('Performing reset: restoring qpos/qvel and zeroing commands')
            data.qpos[:] = initial_qpos
            data.qvel[:] = initial_qvel
            # clear commands and history
            cmd.reset()
            mujoco.mj_forward(model, data)
            cmd.reset_requested = False

        idx=frame_idx(count_lowlevel)
        data.qpos[:] = qpos_list[idx]
        mujoco.mj_fwdPosition(model, data)
        viewer.render()

        elapsed_real_time = time.time() - start_time
        target_sim_time = (step + 1) * cfg.sim_config.dt
        if elapsed_real_time < target_sim_time:
            time.sleep(target_sim_time - elapsed_real_time)

        count_lowlevel += 1

    else:
        viewer.close()
    keyboard_listener.stop()
     # --- Plotting Section (Using only low-frequency data) ---

    print("Simulation finished.")

    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--motion_file',type=str,help='path to motion file(csv)')
    parser.add_argument('--loop',action="store_true",help='loop the policy')
    args = parser.parse_args()

    class Sim2simCfg():

        class sim_config:
            mujoco_model_path = f'{ISAAC_DATA_DIR}/robots/roboparty/atom01/mjcf/atom01.xml'
            sim_duration = 1000.0
            dt = 0.02
            decimation = 1

        class robot_config:
            num_actions = 23
            # 'left_thigh_yaw_joint', 'right_thigh_yaw_joint', 'torso_joint', 'left_thigh_roll_joint', 'right_thigh_roll_joint', 'left_arm_pitch_joint', 'right_arm_pitch_joint', 'left_thigh_pitch_joint', 'right_thigh_pitch_joint', 'left_arm_roll_joint', 'right_arm_roll_joint', 'left_knee_joint', 'right_knee_joint', 'left_arm_yaw_joint', 'right_arm_yaw_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_elbow_pitch_joint', 'right_elbow_pitch_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_elbow_yaw_joint', 'right_elbow_yaw_joint'
            usd2urdf = [0, 6, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22]

    run_mujoco(Sim2simCfg(),args.loop,args.motion_file)
