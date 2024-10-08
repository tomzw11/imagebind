
# imagebind

## run demo
> python demo.py

## todo
- [x] 模型脚本迁移 100%
- [ ] 推理脚本迁移 40%
	- [x] text
	- [x] image
	- [ ] video预处理(torchvideo组件)
	- [ ] audio预处理(torchaudio组件)
- [ ] 交叉模态encode精度对比 15%
	- [ ] text x image
	- [ ] text x audio
	- [ ] image x audio
- [ ] 性能对比

## dev notes/torch diff

### transformer.py
#### Attention
		(mindcv.models.vit)
#### Mlp
#### MultiheadAttention
#### VitAttention
#### BlockWithMasking
		1) 新增droppath(mindcv.models.vit)
#### SimpleTransformer
		1) torch.linspace替换为np.linspace(ms算子替换？)
		_init_weights
			1) 只迁移分支weight_init_style=pytorch
			2) 新增trunc_normal_, constant_(mindone.models.dit)
### helpers.py
#### Normalize
		后处理模块norm，需要单独迁移
		adapted from mindone https://github.com/mindspore-lab/mindone/blob/master/examples/stable_diffusion_xl/gm/modules/util.py
#### LearnableLogitScaling
#### EinOpsRearrange
		1) 待确认是否迁移
#### VerboseNNModule
#### cast_if_src_dtype
#### QuickGELU
#### SelectElement
#### SelectEOSAndProject
#### trunc_normal_（mindone.models.utils）
#### constant_(mindone.models.utils）
#### normal_(mindone.models.utils)
#### zeros_(mindone.models.utils)
#### DropPath
		adapted from mindone droppath module
		to activate need self.dropout.training=True
### multimodal_preprocess.py
#### get_sinusoid_encoding_table
		返回的torch.FloatTensor替换待确认
#### interpolate_pos_encoding_2d
		ops.interpolate（mode=bicubic）可能会有误差
		原实现转换了fp16->fp32->bf16精度，待确认是否迁移
#### interpolate_pos_encoding
#### _get_pos_embedding
#### PatchEmbedGeneric
#### SpatioTemporalPosEmbeddingHelper
		pos_embed用parameter(requires_grad=False)替代register_buffer
#### RGBDTPreprocessor
		去掉了init_parameters()的no_grad装饰器
		用mindone normal_替代nn.init
#### AudioPreprocessor
#### ThermalPreprocessor
#### build_causal_attention_mask
		torch.empty用ops.zeros替代，反正之后会被init覆盖
		用mint.triu替代torch.triu
#### TextPreprocessor
		causal_masking用parameter(requires_grad=False)替代register_buffer
		torch.empty用ops.zeros替代，反正之后会被init覆盖
		去掉了init_parameters()的no_grad装饰器
#### Im2Video
#### PadIm2Video
		用ops.pad替代, 确认args正确
#### bytes_to_unicode
#### get_pairs
#### basic_clean
#### whitespace_clean
#### SimpleTokenizer
#### IMUPreprocessor
		torch.empty用ops.zeros替代，反正之后会被init覆盖
		去掉了init_parameters()的no_grad装饰器
### data.py
	waveform2melspec
		需要替代torchaudio.compliance.kaldi.fbank
		ms.dataset.audio.melscale_fbank对标的是torchaudio的另一个接口torchaudio.functional.melscale_fbank
		两个接口有一定区别，但在torchaudio官网有issue尝试过对齐，待确认
	get_clip_timepoints
	load_and_transform_vision_data
		需要替代pytorchvideo的两个API：
		pytorchvideo.transforms.ShortSideScale
		pytorchvideo.transforms.ConstantClipsPerVideoSampler
	load_and_transform_thermal_data
	load_and_transform_text
	load_and_transform_audio_data
	get_clip_timepoints
	crop_boxes
	uniform_crop
		ops.interpolate（mode=bilinear）可能会有误差
	SpatialCrop
	load_and_transform_video_data
### imagebind_model.py
#### ImageBindModel
		_create_modality_trucks
		_create_modality_heads
		_create_modality_postprocessors
			nn.celllist不支持保存sequentialcelllist,已经解耦preprocessor/head/postprocessor
#### imagebind_huge