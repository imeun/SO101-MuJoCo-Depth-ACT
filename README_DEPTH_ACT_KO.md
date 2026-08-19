# SO-101 MuJoCo Dual-Depth ACT 프로젝트

최종 갱신: 2026-08-19

이 문서는 현재 유지하는 코드의 기준 문서다. 이전의 단일 Depth Student, absolute-action Student, 중간 pose mapping 방식은 폐기했다. 현재 파이프라인은 **MuJoCo scripted teacher -> dual-depth ACT 데이터셋 -> ResNet18 + Transformer ACT 학습 -> MuJoCo 검증 -> SO-101 실로봇 배포**다.

## 1. 현재 목표와 상태

목표는 SO-101이 Top/Side RealSense D435 Depth와 현재 관절 상태를 이용해 세로 블록을 잡고, 사각 상자에 넣고, 집게를 놓은 뒤 초기 자세로 돌아오게 만드는 것이다.

현재 확인된 상태:

- MuJoCo 환경, 카메라, 블록/상자, scripted teacher full-cycle 동작은 구현되어 있다.
- scripted teacher 궤적을 sim-to-real mapping으로 실제 팔에 실행했을 때 동작을 확인했다.
- ACT 체크포인트 `epoch_0008.pt`는 사용자가 확인한 MuJoCo 3회 중 2회 성공했다.
- 학습 중 rollout 로그의 success가 0인 경우가 있었는데, 수동 확인과 평가 seed/조건이 달랐기 때문이다. 체크포인트 품질은 같은 seed 여러 개로 별도 평가해야 한다.
- 학습된 ACT 정책은 실제 로봇에서 아직 안정적으로 성공하지 못했다.
- 관절 mapping보다 **실제 Depth 영상과 MuJoCo Depth의 카메라 외부 파라미터, 결측 픽셀, 배경, 물체 윤곽 차이**가 현재 가장 큰 sim-to-real 문제로 판단된다.
- 노트북 폴더에는 데이터셋, 학습 체크포인트, `realsense_profiles/*_fixed_v3.json`이 없다. 해당 최신 산출물은 Ubuntu 데스크톱에 보관되어 있으므로 별도 백업해야 한다.

### 데모 영상

- [동작 영상 1](media/KakaoTalk_20260812_121136885.mp4)
- [동작 영상 2](media/KakaoTalk_20260812_121154524.mp4)
- [동작 영상 3](media/KakaoTalk_20260812_122736994.mp4)
- [동작 영상 4](media/KakaoTalk_20260812_122920981.mp4)

동영상은 모두 GitHub 단일 파일 제한 100MB보다 작아 저장소에 직접 포함한다. 학습 데이터셋과 체크포인트는 용량이 크므로 `.gitignore`로 제외한다.

## 2. 실제 환경 기준값

- Follower 포트: `/dev/ttyACM0`
- Robot ID: `my_awesome_follower_arm`
- Top D435 serial: `138422072965`
- Side D435 serial: `047322070492`
- 블록 크기: `35 x 35 x 70 mm`
- 상자 내부: `70 x 70 mm`
- 상자 벽 높이: `75 mm`
- 상자 벽 두께: 약 `2 mm`
- 기준 블록 위치: 로봇 앞 `320 mm`, 왼쪽 `155 mm`
- 기준 상자 위치: 로봇 앞 `310 mm`, 오른쪽 `50 mm`
- 최신 노트북 mapping: `sim2real_joint_map_swept.json`

좌표계는 로봇 앞쪽이 `+X`, 로봇 왼쪽이 `+Y`다. 실제 배치나 카메라 장착이 달라지면 데이터 재수집보다 먼저 XML과 실제 외부 파라미터를 다시 맞춰야 한다.

## 3. 최종 폴더 구조

### MuJoCo와 teacher

- `assets/`: SO-101 mesh와 texture.
- `scene_ball_bins.xml`: 테이블, 벽, 블록, 상자, Top/Side D435 카메라가 포함된 최종 scene.
- `so101_new_calib.xml`: SO-101 body, joint, actuator 정의. `scene_ball_bins.xml`이 include하므로 삭제하면 안 된다.
- `so101_ball_bins_env.py`: Gymnasium/MuJoCo 환경, reset/step, 관절·파지·성공 판정.
- `so101_workspace.py`: 블록 위치 샘플링 범위와 workspace 설정.
- `play_waypoint_teacher.py`: open부터 home_hold까지 scripted waypoint teacher 실행과 시각화.
- `teacher_dataset.py`: 성공 teacher 데이터셋 manifest와 episode 저장/검증.
- `collect_scripted_teacher_depth_dataset.py`: 성공 teacher episode를 수집한다.

### ACT 데이터

- `delta_depth_dataset.py`: 최종 ACT 데이터 스키마, phase, control rate, manifest, episode writer.
- `collect_scripted_teacher_delta_depth_dataset.py`: 성공 teacher seed를 다시 실행해 Top/Side Depth와 관절 정답을 동기화한다.
- `collect_fixed_delta_depth_dataset.py`: teacher 생성과 ACT 데이터 저장을 한 번에 수행하는 대안 collector.
- `depth_act_dataset.py`: episode split, frame sample, future action chunk, episode-safe batch sampler.
- `play_fixed_delta_episode.py`: 저장한 정답 episode 재생.
- `visualize_fixed_delta_diversity.py`: 여러 episode의 초기 배치와 궤적 다양성 비교.

### 모델과 학습

- `so101_depth.py`: Depth resize, 정규화, RealSense형 noise augmentation, MuJoCo Depth renderer.
- `so101_depth_act.py`: 1-channel ResNet18, Transformer ACT, loss, temporal ensemble, checkpoint loader.
- `train_depth_act.py`: 학습, validation, MuJoCo rollout 평가, epoch checkpoint 저장.
- `play_depth_act.py`: ACT 체크포인트를 MuJoCo에서 시각적으로 실행.
- `diagnose_depth_act.py`: 데이터 target, zero baseline, prediction, closed-loop 동작 진단.

### DAgger

- `so101_depth_act_dagger.py`: 정책과 waypoint teacher 사이 intervention 판단.
- `collect_depth_act_dagger.py`: MuJoCo에서 실패/불안정 구간을 teacher가 교정한 추가 데이터 수집.

### RealSense와 sim-to-real

- `preview_realsense_depth.py`: 실제 D435 Depth 화면과 serial 확인.
- `measure_realsense_depth_noise.py`: 고정 장면 Depth 통계와 noise profile 생성.
- `sim2real_joint_mapping.py`: LeRobot 관절값과 MuJoCo radian 변환 공통 함수.
- `sweep_joint_calibration.py`: 여러 자세를 사용한 affine joint mapping 생성.
- `sim2real_joint_map_swept.json`: 현재 노트북의 최신 완성 mapping.
- `my_awesome_follower_arm.json`: LeRobot follower calibration 원본.
- `mirror_sim_real.py`: 실제 관절과 MuJoCo 자세 비교 및 명령 제한.
- `tune_elbow_offset_with_robot.py`: 숫자키로 1~5번 관절을 선택해 offset을 미세 조정.
- `run_waypoint_on_real_robot.py`: scripted teacher 정답 궤적을 실제 팔에 안전하게 실행.
- `run_depth_act_model_on_real_robot.py`: 두 D435, ACT, MuJoCo mirror, 안전 제한, Depth 표시를 통합한 최종 배포 파일.
- `tests/`: 현재 파이프라인 회귀 테스트.

## 4. 데이터에 저장되는 값

ACT episode의 제어 주기는 MuJoCo timestep `0.002 s x 17`이라 약 `29.41 Hz`다. 각 제어 시점마다 다음 값을 저장한다.

- `top_depth_mm`: `uint16 [T, 240, 320]`
- `side_depth_mm`: `uint16 [T, 240, 320]`
- `joint_pos`: MuJoCo 6개 관절 위치, radian, `float32 [T, 6]`
- `joint_velocity`: 6개 관절 속도, rad/s, `float32 [T, 6]`
- `teacher_goal_pos`: 그 시점의 teacher 절대 관절 목표, radian
- `delta_target_rad`: `teacher_goal_pos - joint_pos`
- `clean_action`과 `executed_action`: teacher 명령과 실제 적용 명령
- `phase_id`: open, pregrasp, grasp, close, hold, lift, bin, release, settle, retreat, home, home_hold
- `timestamp_s`
- seed, 블록 위치, 상자 위치, 초기 관절 자세, scene/camera hash

D435는 `640 x 480` Depth를 사용하지만 모델 입력 전 nearest-neighbor로 `320 x 240`으로 줄인다. RGB는 사용하지 않는다. Depth는 meter로 변환한 뒤 `0.20~1.00 m` 범위로 clip/normalize한다.

## 5. 모델 입력, 출력, action chunk

한 번의 모델 호출 입력:

- 현재 Top Depth 1장: `[1, 240, 320]`
- 현재 Side Depth 1장: `[1, 240, 320]`
- 현재 6개 관절 위치
- 현재 6개 관절 속도

과거 8프레임, phase ID, 미래 이미지는 모델 입력이 아니다. 현재 관절 위치와 속도 12개가 proprioception이다.

모델 구조:

1. 같은 1-channel ResNet18 backbone이 Top/Side Depth를 각각 feature token으로 변환한다.
2. camera embedding으로 Top과 Side를 구분한다.
3. 관절 위치·속도 12개를 별도 MLP token으로 변환한다.
4. Transformer encoder가 두 카메라와 proprioception을 통합한다.
5. Transformer decoder의 action query 30개가 미래 30개 관절 Delta를 출력한다.

출력 shape은 `[30, 6]`이다. 약 1초 분량의 미래 목표를 한 번에 예측하는 것이 action chunk다. 실행할 때 매 제어 프레임 새 chunk를 만들고, 이전 chunk와 현재 chunk가 예측한 같은 미래 시점의 값을 temporal ensemble로 합친다.

## 6. 정답과 loss

현재 프레임 관절 위치를 `q_t`, 미래 teacher 목표를 `g_(t+k)`라 하면 k번째 정답은 다음과 같다.

```text
target_delta(t, k) = g(t+k) - q(t),  k = 0 ... 29
```

즉, 단순한 다음 프레임 위치 복사가 아니라 현재 자세에서 앞으로 약 1초 동안 도달해야 할 누적 Delta chunk를 학습한다.

전체 loss:

```text
total_loss = weighted_smooth_l1(delta_prediction, delta_target)
           + 0.1 * smooth_l1(predicted_chunk_velocity, target_chunk_velocity)
```

적용 내용:

- 관절별 Delta scale로 정규화한다.
- 큰 움직임 frame은 정지 frame보다 높은 weight를 받는다.
- gripper loss weight는 `1.5`다.
- episode 끝의 padding은 mask로 loss에서 제외한다.
- chunk 내부 변화량 loss를 더해 미래 관절 목표가 갑자기 꺾이는 것을 줄인다.
- `zero_delta_baseline`은 모두 0을 출력하는 모델의 기준 loss일 뿐 실제 loss에 더하지 않는다.
- `zero_ratio = validation_loss / zero_delta_baseline`가 1보다 작아야 하지만, 최종 기준은 closed-loop grasp/lift/success다.

loss가 `0.00001`처럼 작다는 사실만으로 성공 모델은 아니다. 정규화된 radian Delta에 대한 평균 loss이므로 숫자 크기와 실제 task 성공은 직접 대응하지 않는다.

## 7. Ubuntu 기준 전체 실행 순서

### 7.1 환경 변수

```bash
cd /home/zeta/soarm_desktop_fixed
conda activate soarm-mujoco

export PROJECT=/home/zeta/soarm_desktop_fixed
export TEACHER_ROOT="$PROJECT/datasets/near_position_teacher_500_v1"
export ACT_DATASET="$PROJECT/datasets/near_position_depth_act_500_v1"
export ACT_RUN="$PROJECT/runs/near_position_depth_act_500_v1"
export PROFILE_DIR="$PROJECT/realsense_profiles"
```

### 7.2 teacher 동작 먼저 보기

```bash
python play_waypoint_teacher.py \
  --seed 10000 \
  --cube-jitter 0.020 \
  --randomize-bin \
  --bin-jitter 0.010
```

open부터 home_hold까지 블록을 파지, 이동, release하고 초기 자세로 돌아오는지 확인한다.

### 7.3 성공 teacher 500개 생성

```bash
python collect_scripted_teacher_depth_dataset.py \
  --episodes 500 \
  --out-dir "$TEACHER_ROOT" \
  --curriculum near \
  --cube-jitter 0.020 \
  --randomize-bin \
  --bin-jitter 0.010 \
  --capture-stride 1 \
  --max-steps 1100 \
  --seed 10000 \
  --max-attempts 2500
```

고정 초기 관절 자세를 유지하면서 블록은 기준점에서 x/y 각각 약 `+-20 mm`, 상자는 약 `+-10 mm` 범위에서 달라진다. 같은 명령을 500번 복제하는 것이 아니라 각 scene에 맞춰 waypoint/IK를 다시 계산한다.

### 7.4 dual-depth ACT 데이터로 변환

```bash
python collect_scripted_teacher_delta_depth_dataset.py \
  --source-dataset "$TEACHER_ROOT" \
  --out-dir "$ACT_DATASET" \
  --max-episodes 500
```

완료 결과에서 다음을 확인한다.

```text
saved_episodes: 500
capture_stride: 1
initial_conditions: near_position
failed_replays: []
```

### 7.5 정답 episode와 다양성 확인

```bash
python play_fixed_delta_episode.py \
  --dataset "$ACT_DATASET" \
  --episode-index 0 \
  --view-camera free \
  --playback-speed 1.0

python visualize_fixed_delta_diversity.py \
  --dataset "$ACT_DATASET" \
  --episode-indices 0 100 200 300 499 \
  --output "$ACT_DATASET/diversity.png" \
  --open
```

파지 실패, 바닥 관통, 상자 벽 관통, release 전 종료, home 복귀 누락이 있으면 학습 전에 데이터를 고친다.

### 7.6 RealSense noise profile

실제 카메라가 Ubuntu에 연결된 상태에서 고정된 작업 장면을 300프레임 측정한다.

```bash
mkdir -p "$PROFILE_DIR"

python measure_realsense_depth_noise.py \
  --serial 138422072965 \
  --output "$PROFILE_DIR/top_fixed_v3.json" \
  --frames 300 --width 640 --height 480 --fps 30

python measure_realsense_depth_noise.py \
  --serial 047322070492 \
  --output "$PROFILE_DIR/side_fixed_v3.json" \
  --frames 300 --width 640 --height 480 --fps 30
```

profile은 학습 데이터 파일을 다시 쓰지 않는다. 학습 loader가 clean MuJoCo Depth를 읽을 때 frame bias, scale 변화, invalid pixel, edge dropout, hole 형태의 augmentation 범위를 적용한다.

### 7.7 ACT 학습

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python train_depth_act.py \
  --dataset "$ACT_DATASET" \
  --output-dir "$ACT_RUN" \
  --epochs 30 \
  --batch-size 8 \
  --learning-rate 0.0001 \
  --weight-decay 0.0001 \
  --chunk-size 30 \
  --d-model 256 \
  --nhead 8 \
  --encoder-layers 4 \
  --decoder-layers 2 \
  --dim-feedforward 1024 \
  --backbone-width 64 \
  --dropout 0.1 \
  --patience 8 \
  --device cuda \
  --num-workers 4 \
  --log-interval 100 \
  --top-noise-profile "$PROFILE_DIR/top_fixed_v3.json" \
  --side-noise-profile "$PROFILE_DIR/side_fixed_v3.json" \
  --rollout-eval-episodes 3 \
  --rollout-max-steps 1100 \
  --rollout-seed 30000 \
  --rollout-cube-jitter 0.020 \
  --rollout-bin-jitter 0.010
```

저장 파일:

- `checkpoints/epoch_XXXX.pt`: epoch별 원본
- `best_checkpoint.pt`: validation loss 최저
- `best_rollout_checkpoint.pt`: rollout의 success, lift, grasp 순으로 최고
- `last_checkpoint.pt`: 마지막 epoch

### 7.8 MuJoCo에서 체크포인트 확인

```bash
CUDA_VISIBLE_DEVICES="" python play_depth_act.py \
  --model "/home/zeta/soarm_desktop_fixed/runs/near_position_depth_act_500_v1/checkpoints/epoch_0008.pt" \
  --seed 30000 \
  --episodes 3 \
  --max-steps 1100 \
  --cube-jitter 0.020 \
  --bin-jitter 0.010 \
  --ensemble-decay 0.08 \
  --device cpu
```

한 seed 성공만 보지 말고 training과 겹치지 않는 seed 여러 개에서 grasp, lift, release, home을 모두 본다.

### 7.9 진단

```bash
python diagnose_depth_act.py data \
  --model "$ACT_RUN/checkpoints/epoch_0008.pt" \
  --dataset "$ACT_DATASET" \
  --device cuda

python diagnose_depth_act.py onpolicy \
  --model "$ACT_RUN/checkpoints/epoch_0008.pt" \
  --episodes 10 \
  --seed-start 30000 \
  --max-steps 1100 \
  --cube-jitter 0.020 \
  --bin-jitter 0.010 \
  --device cuda
```

validation loss뿐 아니라 예측 Delta 크기, zero baseline, phase별 오차, closed-loop 결과를 함께 본다.

## 8. MuJoCo DAgger 보완

기본 모델이 일부 seed에서 실패할 때만 수행한다.

```bash
python collect_depth_act_dagger.py \
  --policy "$ACT_RUN/best_rollout_checkpoint.pt" \
  --out-dir "$PROJECT/datasets/near_position_depth_act_dagger_r1" \
  --episodes 100 \
  --seed 40000 \
  --max-attempts 500 \
  --cube-jitter 0.020 \
  --bin-jitter 0.010 \
  --trigger-threshold 0.35 \
  --release-threshold 0.12 \
  --minimum-teacher-steps 20 \
  --device cuda \
  --visualize
```

재학습 시 기존 데이터에 correction 데이터를 추가한다.

```bash
python train_depth_act.py \
  --dataset "$ACT_DATASET" \
  --additional-dataset "$PROJECT/datasets/near_position_depth_act_dagger_r1" \
  --additional-repeat 3 \
  --pretrained-checkpoint "$ACT_RUN/best_rollout_checkpoint.pt" \
  --output-dir "$PROJECT/runs/near_position_depth_act_dagger_r1" \
  --epochs 15 \
  --batch-size 8 \
  --device cuda \
  --top-noise-profile "$PROFILE_DIR/top_fixed_v3.json" \
  --side-noise-profile "$PROFILE_DIR/side_fixed_v3.json"
```

## 9. 실제 로봇 적용

### 9.1 카메라 확인

```bash
conda activate lerobot
cd /home/zeta/soarm_desktop_fixed

python preview_realsense_depth.py --list
python preview_realsense_depth.py --serial 138422072965
python preview_realsense_depth.py --serial 047322070492
```

Top/Side가 뒤바뀌지 않았는지, 블록·상자·집게가 Depth에 보이는지, invalid 영역이 과도하지 않은지 확인한다.

### 9.2 scripted waypoint로 mapping 검증

정책 실행 전에 scripted teacher 궤적과 실제 팔이 계속 맞는지 낮은 속도로 확인한다.

```bash
python run_waypoint_on_real_robot.py \
  --calibration "/home/zeta/soarm_desktop_fixed/sim2real_joint_map_swept.json" \
  --port /dev/ttyACM0 \
  --robot-id my_awesome_follower_arm \
  --seed 10000 \
  --playback-speed 0.5 \
  --send
```

실제 동작 옵션은 해당 스크립트의 `--help`와 confirmation을 다시 확인한다.

### 9.3 ACT dry-run과 시각화

`--send`가 없으면 motor command를 보내지 않는다.

```bash
python run_depth_act_model_on_real_robot.py \
  --model "/home/zeta/soarm_desktop_fixed/runs/near_position_depth_act_500_v1/checkpoints/epoch_0008.pt" \
  --calibration "/home/zeta/soarm_desktop_fixed/sim2real_joint_map_swept.json" \
  --port /dev/ttyACM0 \
  --robot-id my_awesome_follower_arm \
  --top-serial 138422072965 \
  --side-serial 047322070492 \
  --device cuda \
  --max-steps 300 \
  --slew 1.5 \
  --align-slew 0.35 \
  --ensemble-decay 0.08 \
  --display \
  --display-rate 5
```

반드시 다음을 확인한다.

- `MuJoCo scene/camera provenance: MATCH`
- `RealSense/MuJoCo intrinsics: MATCH`
- Top/Side serial과 영상 방향
- 현재 실제 관절을 mapping한 MuJoCo mirror 자세
- `max_delta`, loop overrun, tracking ratio가 안전 범위
- 블록 방향으로 일관된 target이 나오는지

### 9.4 실제 구동

주변을 비우고 전원 차단 수단을 준비한 다음에만 `--send`를 추가한다.

```bash
python run_depth_act_model_on_real_robot.py \
  --model "/home/zeta/soarm_desktop_fixed/runs/near_position_depth_act_500_v1/checkpoints/epoch_0008.pt" \
  --calibration "/home/zeta/soarm_desktop_fixed/sim2real_joint_map_swept.json" \
  --port /dev/ttyACM0 \
  --robot-id my_awesome_follower_arm \
  --top-serial 138422072965 \
  --side-serial 047322070492 \
  --device cuda \
  --max-steps 1100 \
  --slew 1.5 \
  --gripper-slew 6.0 \
  --align-slew 0.35 \
  --ensemble-decay 0.08 \
  --display \
  --display-rate 5 \
  --send
```

첫 실험은 block/bin 없이 정렬 동작만 확인하고, 그다음 낮은 속도와 짧은 step으로 진행한다. STOP 창, Space, Esc 또는 Ctrl+C 이후에는 실제 torque 상태를 직접 확인한다.

## 10. 테스트

노트북:

```bat
cd /d D:\IEG\캡스톤\mujoco\soarm
C:\Users\imeun\miniconda3\Scripts\conda.exe run -n mujoco python -m unittest discover -s tests -p "test_*.py" -v
```

Ubuntu:

```bash
cd /home/zeta/soarm_desktop_fixed
conda activate soarm-mujoco
python -m unittest discover -s tests -p 'test_*.py' -v
```

실로봇과 카메라가 없는 환경에서는 hardware integration 자체는 실행하지 않고 변환, safety monitor, 모델, 데이터 테스트만 수행한다.

## 11. 현재 남은 핵심 과제

1. 실제 Top/Side camera extrinsics를 checkerboard/AprilTag 또는 수동 correspondence로 측정해 MuJoCo camera pose와 맞춘다.
2. 같은 실제 장면과 MuJoCo 장면의 Depth histogram, valid pixel 비율, edge map을 수치로 비교한다.
3. 실제 Depth를 MuJoCo 입력 preprocessing에 통과시킨 결과를 저장하고 checkpoint feature 분포를 비교한다.
4. camera mismatch가 줄어든 뒤에만 실로봇 DAgger나 실제 correction 데이터를 추가한다.
5. 실제 scene의 블록/상자 좌표를 Depth로 추정해 MuJoCo digital twin 위치를 갱신하는 방식은 다음 단계다. 현재 코드는 intrinsics만 검사하며 물체 좌표 자동 동기화는 아직 구현하지 않았다.

## 12. 백업할 최신 산출물

노트북 코드 폴더 외에 Ubuntu에서 다음을 반드시 백업한다.

- `datasets/near_position_teacher_500_v1/`
- `datasets/near_position_depth_act_500_v1/`
- `runs/near_position_depth_act_500_v1/`
- `realsense_profiles/top_fixed_v3.json`
- `realsense_profiles/side_fixed_v3.json`
- Ubuntu에서 추가 조정한 최신 mapping이 있다면 `sim2real_joint_map_swept_final.json`

노트북에 있는 `sim2real_joint_map_swept.json`보다 Ubuntu의 `*_final.json`이 실제로 더 최신이면 두 파일을 비교한 뒤 final 파일을 노트북에도 복사하고 모든 실행 명령의 calibration 경로를 final로 통일한다.
