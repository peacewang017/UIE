# Underwater Image Enhancement

Implementation and extension of Underwater Image Enhancement and Attenuation Restoration Based on Depth and Backscatter Estimation (UIBAER) at 
https://ieeexplore.ieee.org/document/10896810.

#### 1. Setup
1. Set up dependencies:
  ```shell
  bash virtual_env_setup.txt
  ``` 

#### 2. Usage

1.  Place desired inputs images in `InputImages` folder.
2.  Run end-to-end implementation:
      ```shell
      python test2.py
      ```
3. Find output images in `OutputImages`.

#### 3. Configuration
Additional approaches to LSAC, Depth Map, Physics models can be found in LSAC_Approaches (isotropic, geometry aware, guided filter, joint intensity/depth regularizer, etc.), Depth_Map_Variations (Scharr, Prewitt, Sobel), Physical_Model_Variations (retinix, etc.)
To swap into pipeline:
1. Copy desired approach from the subfolder to this folder
2. in test.py edit the subprocess to call the submodule 
      Swapping depth map to sobel:
      subprocess.run([sys.executable, "newestdepth.py", filepath], check=True) -> subprocess.run([sys.executable, "depth_sobel.py", filepath], check=True)

      Swapping LSAC to Luma Joint Guided:
      subprocess.run([sys.executable, "LSAC2.py", filepath], check=True) -> subprocess.run([sys.executable, "LSAC_luma_joint_guided.py", filepath], check=True)

#### 4. Error Metrics
Error metrics can be computed using error_uiqm_uciqe_txt.py in OutputImages folder