import data
import mindspore as ms
from mindspore import ops
from imagebind_model import imagebind_huge
from imagebind_model import ModalityType

text_list=["A dog.", "A car", "A bird"]
image_paths=[".assets/dog_image.jpg", ".assets/car_image.jpg", ".assets/bird_image.jpg"]
audio_paths=[".assets/dog_audio.wav", ".assets/car_audio.wav", ".assets/bird_audio.wav"]

ms.set_context(device_target="CPU")

# Instantiate model
model = imagebind_huge(pretrained=True)[0]
model = mode.set_train(mode=False)

# Load data
inputs = {
    ModalityType.TEXT: data.load_and_transform_text(text_list),
    ModalityType.VISION: data.load_and_transform_vision_data(image_paths),
    # ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths),
}

embeddings = model(inputs)

print(
    "Vision x Text: ",
    ops.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, axis=-1),
)
# print(
#     "Audio x Text: ",
#     ops.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, axis=-1),
# )
# print(
#     "Vision x Audio: ",
#     ops.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, axis=-1),
# )

# Expected output:
#
# Vision x Text:
# tensor([[9.9761e-01, 2.3694e-03, 1.8612e-05],
#         [3.3836e-05, 9.9994e-01, 2.4118e-05],
#         [4.7997e-05, 1.3496e-02, 9.8646e-01]])
#
# Audio x Text:
# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]])
#
# Vision x Audio:
# tensor([[0.8070, 0.1088, 0.0842],
#         [0.1036, 0.7884, 0.1079],
#         [0.0018, 0.0022, 0.9960]])
