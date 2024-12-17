# -*- coding: utf-8 -*-
from argument import args
from synthesize.main_0 import main as main_0
from synthesize.main_GUIDE import main as main_Guide
from synthesize.main_GUIDE_schedule import main as main_Guide_schedule
from synthesize.main_2 import main as main_2
from synthesize.main_1 import main as main_1
import cProfile
# from synthesize.main_02 import main as main_02
from validation.main_3 import (
    main as main_3,
)  # The relabel and validation are combined here for fast experiment
import warnings
warnings.filterwarnings("ignore")

if __name__  == "__main__":
    #main_0(args) #시작점이 되는 랜덤한 이미지를 원본 데이터 셋에서 고름
    #main_Guide(args) #Classifier Guidance를 활용한 Diffusion
    main_2(args) # RDED after DIF | Diffusion 으로 생성한 이미지들을 RDED에서 쓰던 crop 후 teacher model에서 evaluate해서 고름
    main_3(args) #Evaluation | Distill을 해서 생성된 이미지를 최종 평가