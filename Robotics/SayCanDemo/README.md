## Cliport: CLIP conditioned transportNet
#### ClipModel, CliportVariant, CliportTraning, CliportDemo
#### Saycan>Demo>Runner.py: 
    *ViLD* used to find objects == Affordance scoring
    *Cliport* used to find pick and place position
##### CliportDemo:
    run_cliport: cal action (pick/place pos) and 
    interact with env.

#### 

#### Requirements: 
1. See requirements.txt 
2.download assets:
2.1 If you are using GooglColab
  #Download PyBullet assets.
  if not os.path.exists('ur5e/ur5e.urdf'):
    !gdown --id 1Cc_fDSBL6QiDvNT4dpfAEbhbALSVoWcc
    !gdown --id 1yOMEm-Zp_DL3nItG9RozPeJAmeOldekX
    !gdown --id 1GsqNLhEl9dd4Mc3BM0dX3MibOI1FVWNM
    !unzip ur5e.zip
    !unzip robotiq_2f_85.zip
    !unzip bowl.zip

  # ViLD pretrained model weights.
  !gsutil cp -r gs://cloud-tpu-checkpoints/detection/projects/vild/colab/image_path_v2 ./
