# SeqSNN

A public framework for time-series forecasting with spiking neural networks (SNNs).

## Research-Extended Version
이 리포지토리는 Microsoft에서 공개한 [SeqSNN](https://github.com/microsoft/SeqSNN)을 기반으로,
다변량 시계열 예측에서 채널 간 구조적 정보를 스파이크 형태로 통합하기 위한 연구 목적으로 확장된 버전입니다.

### 주요 변경 사항
본 fork 버전에서는 연구 실험을 위해 다음과 같은 기능이 추가 및 수정되었습니다.
* **채널 유사성 기반 스파이크 인코딩(Channel Similarity-based Encoding, CSE)** 모듈 추가
  → 어텐션 기반 채널 클러스터링과 Straight-Through Estimator(STE)를 이용한 시간적 인코딩과의 통합
* **추가 실험 구성 파일(exp/forecast/cluster/)** 제공
* 실험 지원 및 결과 수집 스크립트 제공

본 확장 버전은 **'다변량 시계열 예측을 위한 채널 유사성 기반 스파이크 인코딩 기법'** 연구의 목적이며,
Microsoft 또는 원저자에 의해 공식적으로 유지 및 관리되지 않습니다.

## Related Papers
* Efficient and Effective Time-Series Forecasting with Spiking Neural Networks, [ICML 2024], (https://arxiv.org/pdf/2402.01533).
* Advancing Spiking Neural Networks for Sequential Modeling with Central Pattern Generators, [NeurIPS 2024], (https://arxiv.org/pdf/2405.14362).


## Installation
To install SeqSNN in a new conda environment:
```
conda create -n SeqSNN python=[3.8, 3.9, 3.10]
conda activate SeqSNN
git clone https://github.com/pioneer0413/SeqSNN.git
cd SeqSNN
pip install .
```

If you would like to make changes and run your experiments, use:

`pip install -e .`

## Training
Take the `iSpikformer` model as an example:

`python -m SeqSNN.entry.tsforecast exp/forecast/ispikformer/ispikformer_electricity.yml`

You can change the `yml` configuration files as you want.

You can add, remove, or modify your model architecture in `SeqSNN/network/XXX.py`.

## Datasets

연구에 사용된 모든 실험 데이터는 [MTSF-dataset](https://github.com/pioneer0413/MTSF-dataset)에서 획득할 수 있습니다.

The folder structure of this project is as follows:
```
SeqSNN
│   README.md 
│   ...
│
└───data
│   │   ETTh1.txt
│   │   ETTh2.txt
│   │   metr-la.h5
│   │   Weather.txt
│   │   
│   └───solar-energy
│   │   │   solar_AL.txt
│   │   │   ...
│   │   
│   └───electricity
│       │   electricity.txt
│       │   ...
│
└───exp
│   │   ...
│
└───outputs
│   │   ...
│
```
You can change the path of the data file in `exp/forecast/dataset/XXX.yml` configuration files.

## Acknowledgement
This repo is built upon [forecaster](https://github.com/Arthur-Null/SRD), which is a general time-series forecasting library. We greatly thank @rk2900 and @Arthur-Null for their initial contribution. 

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
