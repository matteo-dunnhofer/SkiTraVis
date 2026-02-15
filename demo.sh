VIDEO=videos/video_example.mp4
FEET_POS_FACTOR=0.9
TRAJ_W=10
TRAJ_O=2
ATHLETE="Athlete"

python run_skitravis.py \
    --source $VIDEO \
    --view-img \
    --smooth-trajectory \
    --feet-position-factor $FEET_POS_FACTOR \
    --trajectory-smooth-window $TRAJ_W \
    --trajectory-smooth-order $TRAJ_O \
    --athlete-name $ATHLETE \
    --manual-init 