# [SkiTraVis](https://openaccess.thecvf.com/content/CVPR2023W/CVSports/papers/Dunnhofer_Visualizing_Skiers_Trajectories_in_Monocular_Videos_CVPRW_2023_paper.pdf)

Official code for SkiTraVis — software for ski trajectory visualization and analysis from video sequences.


# Abstract
*Trajectories are fundamental to winning in alpine skiing. Tools enabling the analysis of such curves can enhance the training activity and enrich broadcasting content. In this paper, we propose SkiTraVis, an algorithm to visualize the sequence of points traversed by a skier during its performance. SkiTraVis works on monocular videos and constitutes a pipeline of a visual tracker to model the skier's motion and of a frame correspondence module to estimate the camera's motion. The separation of the two motions enables the visualization of the trajectory according to the moving camera's perspective. We performed experiments on videos of real-world professional competitions to quantify the visualization error, the computational efficiency, as well as the applicability. Overall, the results achieved demonstrate the potential of our solution for broadcasting media enhancement and coach assistance.*


Demo of the capabilities of SkiTraVis are available in this video:

<iframe width="560" height="315" src="https://www.youtube.com/embed/JPQ6EeOaon0?si=e6-qBGYZ2p67AzZu" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Usage

Download the source code and install requirements.

```bash
git clone https://github.com/matteo-dunnhofer/SkiTraVis.git
cd SkiTraVis
```
Install requirements with:
```bash
conda env create -f skitravis.yml
```
Code tested on Ubuntu 22.04+, Python 3.8+, PyTorch 1.7+ (CUDA).


To run SkiTraVis use the run_skitravis.py script.

- Specify input:
    - --source PATH — path to a video file or a folder of frames (supports common video formats and image sequences).
    - Example:
        ```bash
        python run_skitravis.py --source /path/to/video.mp4
        ```

- Default behavior:
    - The skier is automatically detected in the first frame.
    - The output video with the visualized trajectory is saved as an MP4 in the output/ folder (name derived from the input).

- View while processing:
    - Add --view to display the output window during processing:
        ```bash
        python run_skitravis.py --source /path/to/video.mp4 --view
        ```

- Manual initialization:
    - Add --manual-init to draw the initial bounding box yourself:
        ```bash
        python run_skitravis.py --source /path/to/video.mp4 --manual-init
        ```
    - A window opens for selection: click the top-left corner, drag to the bottom-right, then press Enter to confirm (Esc to cancel). Processing starts after confirmation.

An example is given by:
```bash
bash demo.sh
```

## Contact

Feel free to open an issue on GitHub for any problems. Otherwise you can contact me via e-mail by writing to [matteo.dunnhofer@uniud.it](matteo.dunnhofer@uniud.it).

## Reference
If you find this work useful please cite
```
@inproceedings{dunnhofer2023visualizing,
  title={Visualizing Skiers' Trajectories in Monocular Videos},
  author={Dunnhofer, Matteo and Sordi, Luca and Micheloni, Christian},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  pages={5188--5198},
  year={2023}
}

## Contributions

I look forward in collaborating to improve the robustness of SkiTraVis and to make the tool more user-friendly (e.g. easy usable for high-level users such as sport scientists or coaches).


## Acknowledgements

We thank the authors of [STARK](https://github.com/researchmm/Stark), [YOLOv5](https://github.com/ultralytics/yolov5), [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork), and [Kornia](https://github.com/kornia/kornia) for inspiring our work. The code in this repository heavily borrows from their original repositories.

License: MIT