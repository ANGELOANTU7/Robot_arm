# Robot_arm in a container: the simulator, the video pipeline and training.
#   docker build -t robot_arm .
#   docker run --rm -it robot_arm python scripts/g1_evaluate.py --task full --episodes 20
# With an NVIDIA GPU: docker build --build-arg TORCH_INDEX=https://pypi.org/simple -t robot_arm:gpu .
#                     docker run --rm -it --gpus all -e MUJOCO_GL=egl robot_arm:gpu ...
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libgl1 libegl1 libosmesa6 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Off-screen rendering without a display; set MUJOCO_GL=egl when running with a GPU.
ENV MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONUNBUFFERED=1

# CPU-only PyTorch by default (the CUDA build's Triton crashes on import without a GPU); see the GPU build above.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch --index-url ${TORCH_INDEX}

WORKDIR /app
COPY pyproject.toml README.md ./
COPY robot_arm ./robot_arm
RUN pip install --no-cache-dir -e ".[dev,pose]"
COPY . .

# Download the G1, Shadow hand and Panda models from MuJoCo Menagerie into the image.
RUN python -c "from robot_arm.assets import ensure_model as m; m('unitree_g1', 'g1_with_hands.xml'); \
    m('shadow_hand', 'right_hand.xml'); m('franka_emika_panda', 'panda.xml')"

CMD ["bash"]
