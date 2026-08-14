import os
import cv2
import sys
import torch
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy import signal

# 10: head
# 0: center pelvis
# 1,4: pelvis (R, L)
# 2,5: knee
# 3,6: feet
# 11,14: shoulder (R,L)
# 12,15: elbow (R,L)
# 13,16: hand (R,L)
def resize_keypoint(keypoint, length):
    keypoint_new = np.zeros((length, 17, 3))
    t_ori = np.linspace(0,len(keypoint)-1,len(keypoint))
    t_new = np.linspace(0,len(keypoint)-1,length)
    for body in range(17):
        for coord in range(3):
            keypoint_sel = keypoint[:,body,coord]
            keypoint_interp = np.interp(t_new, t_ori, keypoint_sel)
            keypoint_new[:,body,coord] = keypoint_interp
    return keypoint_new

def runningaverage_keypoint(keypoint, beta = 0.9):
    keypoint_new = np.zeros((len(keypoint), 17, 3))
    for body in range(17):
        for coord in range(3):
            for t in range(len(keypoint)):
                if t==0:
                    keypoint_new[t,body,coord] = keypoint[t,body,coord]
                else:
                    keypoint_new[t,body,coord] = beta*keypoint[t,body,coord] + (1-beta)*keypoint_new[t-1,body,coord]
    return keypoint_new

def img2video(img_dir, vid_dir):
    video_name = ('_').join(img_dir.split('/')[-3:-1])
    fps = 30.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    names = sorted(glob.glob(os.path.join(img_dir, '*.png')))
    img = cv2.imread(names[0])
    size = (img.shape[1], img.shape[0])
    videoWrite = cv2.VideoWriter(vid_dir, fourcc, fps, size) 
    for name in names:
        img = cv2.imread(name)
        videoWrite.write(img)
    videoWrite.release()
    return



def show_all(vid, data_GT, data_Pred, data_startpoint, select_frame, path=None):
    #
    path_folder = ('/').join(path.split('/')[:-1])
    path_temp   = os.path.join(path_folder,'temp/')
    os.makedirs(path_temp, exist_ok=True)
    t_idx = np.linspace(data_startpoint[select_frame], data_startpoint[select_frame]+90-1, 90)
    # Keypoint
    output_GT   = data_GT[select_frame,:,:,:]
    output_Pred = data_Pred[select_frame,:,:,:]
    output_GT   = resize_keypoint(output_GT, 90)
    output_Pred = resize_keypoint(output_Pred, 90)
    for i in range(len(t_idx)):
        img       = vid[int(t_idx[i])]
        pose_GT   = output_GT[i]
        pose_Pred = output_Pred[i]
        ## show
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.0, hspace=0.0)
        ax1 = fig.add_subplot(gs[0, 1], projection='3d')
        ax2 = fig.add_subplot(gs[0, 2], projection='3d')
        ax3 = fig.add_subplot(gs[0, 0])
        show3Dpose(pose_GT, ax1, save=False)
        show3Dpose(pose_Pred, ax2, save=False)
        showimage(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), ax3)
        ## save
        plt.savefig(path_temp + str(('%04d'% i)) + '_pose.png', dpi=300, bbox_inches = 'tight')
        fig.tight_layout()
        plt.close()
    img2video(path_temp, path)
    shutil.rmtree(path_temp)
    return

def showRGB_video(vid, data_startpoint, select_frame, path=None):
    fps = 30.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    size = (1920, 1080)
    videoWrite = cv2.VideoWriter(path, fourcc, fps, size)
    t_idx = np.linspace(data_startpoint[select_frame], data_startpoint[select_frame]+90-1, 90)
    for i in t_idx:
        videoWrite.write(vid[int(i)])
    videoWrite.release()
    return

def show3Dpose_fromradar_video(device, data_keypoint, data_keypoint_startpoint, select_frame='all', path=None):
    """
    Input: 
        - data_keypoint: N_sampled keypoints from 10-s episode  (N_sample,T,17,3)
        - data_keypoint_startpoint: Sample t_idx of N_sample data
    """
    # Interpolate from 19 -> 90
    # idx_sel = loss.argmin()
    if select_frame=='all':
        output_sel = np.zeros((10,300,17,3))
        val_num = np.zeros(300)
        for idx_sel in range(len(data_keypoint_startpoint)):
            startpoint = data_keypoint_startpoint[idx_sel]
            output_temp = data_keypoint[idx_sel,:,:,:]
            output_temp = resize_keypoint(output_temp, 90)
            output_sel[idx_sel,startpoint:startpoint+90,:,:] = output_temp
            val_num[startpoint:startpoint+90] += 1.
        output_sel = output_sel.sum(axis=0)
        idx_nonzero = np.where(val_num != 0)[0]
        val_nonzero = np.repeat(val_num[idx_nonzero], 17*3).reshape(-1,17,3)
        output_sel = output_sel[idx_nonzero,:,:]/val_nonzero
        # filter
        output_sel = runningaverage_keypoint(output_sel, beta=0.95)
    else:
        idx_sel = select_frame
        output_sel = data_keypoint[idx_sel,:,:,:]
        output_sel = resize_keypoint(output_sel, 90)
    
    length = len(output_sel)
    fps = 30.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    size = (640, 480)
    # for estimation
    videoWrite = cv2.VideoWriter(path, fourcc, fps, size)
    for i in range(length):
        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        show3Dpose(output_sel[i], ax, save=False)
        # figure to img data
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # write img to video
        videoWrite.write(data)
        plt.close()
    videoWrite.release()
    return


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

def save_academic_video(x_mD, pose_pred_seq, pose_gt_seq, save_path, fps=30, video_frames=90):



    spectrogram = x_mD.detach().cpu().numpy() if torch.is_tensor(x_mD) else x_mD
    while spectrogram.ndim > 2:
        spectrogram = spectrogram[0]
    num_freqs, T_radar = spectrogram.shape 
    

    p_pred = pose_pred_seq.detach().cpu().numpy() if torch.is_tensor(pose_pred_seq) else pose_pred_seq
    p_gt = pose_gt_seq.detach().cpu().numpy() if torch.is_tensor(pose_gt_seq) else pose_gt_seq
    T_pose = p_pred.shape[0]

    def interpolate_pose(pose_seq, target_len):
        orig_len = pose_seq.shape[0]
        joints, dims = pose_seq.shape[1], pose_seq.shape[2]
        new_seq = np.zeros((target_len, joints, dims))
        t_ori = np.linspace(0, 1, orig_len)
        t_new = np.linspace(0, 1, target_len)
        for j in range(joints):
            for d in range(dims):
                new_seq[:, j, d] = np.interp(t_new, t_ori, pose_seq[:, j, d])
        return new_seq

    smooth_pred = interpolate_pose(p_pred, video_frames)
    smooth_gt = interpolate_pose(p_gt, video_frames)


    width, height = 1800, 600 
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    def set_academic_3d_grid(ax):

        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.grid(True)
        ax.xaxis._axinfo["grid"].update({"color": "lightgray", "linewidth": 0.8, "linestyle": "-."})
        ax.yaxis._axinfo["grid"].update({"color": "lightgray", "linewidth": 0.8, "linestyle": "-."})
        ax.zaxis._axinfo["grid"].update({"color": "lightgray", "linewidth": 0.8, "linestyle": "-."})


    plt.ioff()
    fig = plt.figure(figsize=(18, 6), dpi=100)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.8, 1, 1], wspace=0.1)

    for i in range(video_frames):
        fig.clf()
        

        current_radar_t = i * (T_radar - 1) / (video_frames - 1)
        

        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
        ax1.axvline(x=current_radar_t, color='#E53935', linestyle='-', linewidth=3)
        ax1.set_title("Micro-Doppler Spectrogram", fontsize=16, fontweight='bold')
        # ax1.set_ylabel("Doppler Frequency", fontsize=14)
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2 = fig.add_subplot(gs[1], projection='3d')
        show3Dpose(smooth_pred[i], ax2, save=False)
        set_academic_3d_grid(ax2)
        ax2.set_title("Ours (PDNet)", fontsize=16, fontweight='bold', color='#2878B5')
        

        ax3 = fig.add_subplot(gs[2], projection='3d')
        show3Dpose(smooth_gt[i], ax3, save=False)
        set_academic_3d_grid(ax3)
        ax3.set_title("Ground Truth", fontsize=16, fontweight='bold', color='gray')


        fig.canvas.draw()
        rgba_buffer = np.asarray(fig.canvas.buffer_rgba())
        img_bgr = cv2.cvtColor(rgba_buffer, cv2.COLOR_RGBA2BGR)
        

        img_bgr = cv2.resize(img_bgr, (width, height))
        video_writer.write(img_bgr)

    video_writer.release()
    plt.close(fig)



import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def save_paper_visualization_frames(x_mD, x_R, pose_gt_seq, pose_pred_seq, save_dir, prefix=""):

    gt_dir = os.path.join(save_dir, "GT_Poses")
    pred_dir = os.path.join(save_dir, "Pred_Poses")
    spec_md_h_dir = os.path.join(save_dir, "Doppler_Horizontal")
    spec_rng_h_dir = os.path.join(save_dir, "Range_Horizontal")
    
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(spec_md_h_dir, exist_ok=True)
    os.makedirs(spec_rng_h_dir, exist_ok=True)
    

    mD_specs = x_mD.detach().cpu().numpy() if torch.is_tensor(x_mD) else x_mD
    while mD_specs.ndim > 3:
        mD_specs = mD_specs[0]
        

    rng_specs = x_R.detach().cpu().numpy() if torch.is_tensor(x_R) else x_R
    while rng_specs.ndim > 3:
        rng_specs = rng_specs[0]
        

    num_views, _, T_mD = mD_specs.shape 
    _, _, T_Rng = rng_specs.shape 
    T_pose = pose_gt_seq.shape[0]
    
    if num_views > 1:
        spec_md_v_dir = os.path.join(save_dir, "Doppler_Vertical")
        spec_rng_v_dir = os.path.join(save_dir, "Range_Vertical")
        os.makedirs(spec_md_v_dir, exist_ok=True)
        os.makedirs(spec_rng_v_dir, exist_ok=True)
        

    
    for t in range(T_pose):

        p_gt = pose_gt_seq[t].detach().cpu().numpy() if torch.is_tensor(pose_gt_seq[t]) else pose_gt_seq[t]
        p_pred = pose_pred_seq[t].detach().cpu().numpy() if torch.is_tensor(pose_pred_seq[t]) else pose_pred_seq[t]
        

        frame_mpjpe = np.mean(np.linalg.norm(p_pred - p_gt, axis=-1))
        


        file_name = f"frame_{t:02d}_mpjpe_{frame_mpjpe:.6f}.png"

        if T_pose > 1:
            current_mD_t = t * (T_mD - 1) / (T_pose - 1)
            current_rng_t = t * (T_Rng - 1) / (T_pose - 1)
        else:
            current_mD_t = T_mD / 2.0
            current_rng_t = T_Rng / 2.0
        

        fig, ax = plt.subplots(figsize=(8, 3), dpi=300)
        ax.imshow(mD_specs[0], aspect='auto', origin='lower', cmap='viridis')
        ax.axvline(x=current_mD_t, color='#E53935', linestyle='-', linewidth=3)
        ax.axis('off')
        plt.savefig(os.path.join(spec_md_h_dir, file_name), bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        if num_views > 1:
            fig, ax = plt.subplots(figsize=(8, 3), dpi=300)
            ax.imshow(mD_specs[1], aspect='auto', origin='lower', cmap='viridis')
            ax.axvline(x=current_mD_t, color='#E53935', linestyle='-', linewidth=3)
            ax.axis('off')
            plt.savefig(os.path.join(spec_md_v_dir, file_name), bbox_inches='tight', pad_inches=0)
            plt.close(fig)


        cmap_rng = 'plasma'
        
        fig, ax = plt.subplots(figsize=(8, 3), dpi=300)
        ax.imshow(rng_specs[0], aspect='auto', origin='lower', cmap=cmap_rng)
        ax.axvline(x=current_rng_t, color='#00E676', linestyle='-', linewidth=3)
        ax.axis('off')
        plt.savefig(os.path.join(spec_rng_h_dir, file_name), bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        if num_views > 1:
            fig, ax = plt.subplots(figsize=(8, 3), dpi=300)
            ax.imshow(rng_specs[1], aspect='auto', origin='lower', cmap=cmap_rng)
            ax.axvline(x=current_rng_t, color='#00E676', linestyle='-', linewidth=3)
            ax.axis('off')
            plt.savefig(os.path.join(spec_rng_v_dir, file_name), bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            

        def set_academic_3d_grid(ax):

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.grid(True)
            ax.xaxis._axinfo["grid"].update({"color": "lightgray", "linewidth": 0.8, "linestyle": "-."})
            ax.yaxis._axinfo["grid"].update({"color": "lightgray", "linewidth": 0.8, "linestyle": "-."})
            ax.zaxis._axinfo["grid"].update({"color": "lightgray", "linewidth": 0.8, "linestyle": "-."})


        fig = plt.figure(figsize=(4, 4), dpi=300)
        ax = fig.add_subplot(111, projection='3d')
        show3Dpose(pose_gt_seq[t], ax, save=False)
        set_academic_3d_grid(ax)  
        plt.savefig(os.path.join(gt_dir, file_name), bbox_inches='tight', pad_inches=0.1, transparent=True)
        plt.close(fig)
        

        fig = plt.figure(figsize=(4, 4), dpi=300)
        ax = fig.add_subplot(111, projection='3d')
        show3Dpose(pose_pred_seq[t], ax, save=False)
        set_academic_3d_grid(ax)  
        plt.savefig(os.path.join(pred_dir, file_name), bbox_inches='tight', pad_inches=0.1, transparent=True)
        plt.close(fig)


def show3Dpose_video(vals_pre, path=None):
    length = len(vals_pre)
    fps = 30.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    size = (640, 480)
    videoWrite = cv2.VideoWriter(path, fourcc, fps, size)
    for i in range(length):
        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        show3Dpose(vals_pre[i], ax, save=False)
        # figure to img data
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # write img to video
        videoWrite.write(data)
        plt.close()
    videoWrite.release()
    return
	


def showimage(img, ax):
    ax.set_xticks([])
    ax.set_yticks([]) 
    plt.axis('off')
    ax.imshow(img)

def show3Dpose(vals_pre, ax, path=None, save=True):
    ax.view_init(elev=15., azim=70)

    vals_pre = np.array(vals_pre, dtype='float32')
    rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
    rot = np.array(rot, dtype='float32')
    vals = camera_to_world(vals_pre, R=rot, t=0)
    # vals = np.array(vals_pre)

    lcolor=(1,0,0)  # Strange.. Why is this color reversed with 2D? (TODO)
    rcolor=(0,0,1)

    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0,   1,  0,  0,  1,  1, 0, 0], dtype=bool)

    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, color = lcolor if LR[i] else rcolor)
    # test_idx = 1
    # ax.scatter(vals[test_idx,0],vals[test_idx,1],vals[test_idx,2], 'o', color=(0,1,0))
    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS_Z+zroot, RADIUS_Z+zroot])
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    ax.set_aspect('equal') # works fine in matplotlib==2.2.2

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    # remove tick labels
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])

    # ax.tick_params('x', labelbottom = True)
    # ax.tick_params('y', labelleft = True)
    # ax.tick_params('z', labelleft = True)
    if save==True:
        plt.savefig(path, dpi=400, bbox_inches = 'tight')

def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2
    return X / w * 2 - [1, h / w]


def world_to_camera(X, R, t):
    Rt = wrap(qinverse, R) 
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - t) 


def camera_to_world(X, R, t):
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t


def wrap(func, *args, unsqueeze=False):
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)
    result = func(*args)

    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result): 
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result

def qrot(q, v):
	assert q.shape[-1] == 4
	assert v.shape[-1] == 3
	assert q.shape[:-1] == v.shape[:-1]

	qvec = q[..., 1:]
	uv = torch.cross(qvec, v, dim=len(q.shape) - 1)
	uuv = torch.cross(qvec, uv, dim=len(q.shape) - 1)
	return (v + 2 * (q[..., :1] * uv + uuv))


def qinverse(q, inplace=False):
    if inplace:
        q[..., 1:] *= -1
        return q
    else:
        w = q[..., :1]
        xyz = q[..., 1:]
        return torch.cat((w, -xyz), dim=len(q.shape) - 1)



        