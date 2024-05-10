include cmsis.mk
include tflite.mk

OBJ_DIR = build
TARGET = $(OBJ_DIR)/main

ARM_GCC_DIR = /gcc-arm-none-eabi-10-2020-q4-major
FVP_CORSTONE_DIR = /usr/local/FVP_Corstone_SSE-300/models/Linux64_GCC-6.4

AS = $(ARM_GCC_DIR)/bin/arm-none-eabi-as
LD = $(ARM_GCC_DIR)/bin/arm-none-eabi-g++
CXX = $(ARM_GCC_DIR)/bin/arm-none-eabi-g++
CC = $(ARM_GCC_DIR)/bin/arm-none-eabi-gcc
OC = $(ARM_GCC_DIR)/bin/arm-none-eabi-objcopy
OD = $(ARM_GCC_DIR)/bin/arm-none-eabi-objdump
OS = $(ARM_GCC_DIR)/bin/arm-none-eabi-size
RM = rm

INCLUDE_DIRS += $(ARM_GCC_DIR)/arm-none-eabi/include \
	. \
	third_party/flatbuffers/include \
	$(CMSIS_DIR)/Device/ARM/ARMCM55/Include \
	sparse_kernels/Include \

ASOURCES += 

CSOURCES += third_party/cmsis/Device/ARM/ARMCM55/Source/startup_ARMCM55.c \
	third_party/cmsis/Device/ARM/ARMCM55/Source/system_ARMCM55.c \

CSOURCES += $(shell find ./sparse_kernels/Source -type f -name '*.c') \
	$(shell find ./apps -type f -name '*.c') \

CXXSOURCES += $(shell find ./apps -type f -name '*.cc') \

INCLUDE_DIRS += apps \

# Which app to build
ifdef RESNET12_CIFAR10
INCLUDE_DIRS += apps/resnet12_cifar10
CXXSOURCES += $(shell find ./apps/resnet12_cifar10 -type f -name '*.cc')
endif

ifdef VGG11_CIFAR10
INCLUDE_DIRS += apps/vgg11_cifar10
CXXSOURCES += $(shell find ./apps/vgg11_cifar10 -type f -name '*.cc')
endif

ifdef SQUEEZENET_CIFAR10
INCLUDE_DIRS += apps/squeezenet_cifar10
CXXSOURCES += $(shell find ./apps/squeezenet_cifar10 -type f -name '*.cc')
endif

ifdef VGG11_GTSRB
INCLUDE_DIRS += apps/vgg11_gtsrb
CXXSOURCES += $(shell find ./apps/vgg11_gtsrb -type f -name '*.cc')
endif

ifdef RESNET12_GTSRB
INCLUDE_DIRS += apps/resnet12_gtsrb
CXXSOURCES += $(shell find ./apps/resnet12_gtsrb -type f -name '*.cc')
endif

ifdef SQUEEZENET_GTSRB
INCLUDE_DIRS += apps/squeezenet_gtsrb
CXXSOURCES += $(shell find ./apps/squeezenet_gtsrb -type f -name '*.cc')
endif

ifdef VGG11_VWW
INCLUDE_DIRS += apps/vgg11_vww
CXXSOURCES += $(shell find ./apps/vgg11_vww -type f -name '*.cc')
endif

ifdef RESNET12_VWW
INCLUDE_DIRS += apps/resnet12_vww
CXXSOURCES += $(shell find ./apps/resnet12_vww -type f -name '*.cc')
endif

ifdef SQUEEZENET_VWW
INCLUDE_DIRS += apps/squeezenet_vww
CXXSOURCES += $(shell find ./apps/squeezenet_vww -type f -name '*.cc')
endif

ifdef VGG11_CAMERA_CATALOGUE
INCLUDE_DIRS += apps/vgg11_camera_catalogue
CXXSOURCES += $(shell find ./apps/vgg11_camera_catalogue -type f -name '*.cc')
endif

ifdef RESNET12_CAMERA_CATALOGUE
INCLUDE_DIRS += apps/resnet12_camera_catalogue
CXXSOURCES += $(shell find ./apps/resnet12_camera_catalogue -type f -name '*.cc')
endif

ifdef SQUEEZENET_CAMERA_CATALOGUE
INCLUDE_DIRS += apps/squeezenet_camera_catalogue
CXXSOURCES += $(shell find ./apps/squeezenet_camera_catalogue -type f -name '*.cc')
endif

ifdef SINGLE_CONV
INCLUDE_DIRS += apps/single_conv
CXXSOURCES += $(shell find ./apps/single_conv -type f -name '*.cc')
endif

ifdef YOLO
INCLUDE_DIRS += apps/yolo
CXXSOURCES += $(shell find ./apps/yolo -type f -name '*.cc')
endif


DEFS += -DARMCM55 -DCPU_M55=1 -DCORTEX_M_GENERIC
DEFS += -DVALUE_BUFFERING
MCUFLAGS = -mcpu=cortex-m55 -mfloat-abi=hard -mthumb -funsigned-char -mlittle-endian
OPTIMIZE = -O1 -flto

WFLAGS = -Wall -Wextra
ASMFLAGS = $(MCUFLAGS) $(DEFS) $(OPTIMIZE) -fmessage-length=0 -c
CFLAGS = -c $(MCUFLAGS) $(DEFS) $(OPTIMIZE) $(WFLAGS) $(addprefix -I,$(INCLUDE_DIRS)) -std=c11 -ffunction-sections -fdata-sections
CXXFLAGS = -c $(MCUFLAGS) $(DEFS) $(OPTIMIZE) $(WFLAGS) $(addprefix -I,$(INCLUDE_DIRS)) -std=c++11 -ffunction-sections -fdata-sections -fno-rtti -fno-exceptions
LIBS = -lnosys

LSCRIPT = ./gcc_arm.ld

LDFLAGS += $(MCUFLAGS) \
		--entry=Reset_Handler \
		-static -g\
		-Xlinker -Map=$(TARGET).map \
		-T$(LSCRIPT) \
		-Wl,-gc-sections \
		$(OPTIMIZE) \
		--specs=nosys.specs \
		--specs=nano.specs \

OBJECTS = $(addsuffix .o,$(addprefix $(OBJ_DIR)/,$(basename $(ASOURCES))))
OBJECTS += $(addsuffix .o,$(addprefix $(OBJ_DIR)/,$(basename $(CSOURCES))))
OBJECTS += $(addsuffix .o,$(addprefix $(OBJ_DIR)/,$(basename $(CXXSOURCES))))

.PHONY: clean all simulate size clean_apps

all: $(TARGET).axf

$(TARGET).axf: $(TARGET).elf
	cp $< $@

$(TARGET).elf: $(OBJECTS)
	@echo Linking: $@
	$(LD) -o $@ $^ $(LIBS) $(LDFLAGS)
	$(OD) -h -S -C --no-addresses $(TARGET).elf > $(TARGET).lst

size: $(TARGET).elf
	@echo
	@echo == Object size ==
	@$(OS) --format=berkeley $<

$(OBJ_DIR)/%.o: %.cc
	@mkdir -p $(dir $@)
	@echo %% $(notdir $<)
	@$(CXX) -c -o $@ $(CXXFLAGS) $<

$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	@echo %% $(notdir $<)
	@$(CC) -c -o $@ $(CFLAGS) $<

$(OBJ_DIR)/%.o: %.s
	@mkdir -p $(dir $@)
	@echo %% $(notdir $<)
	@$(CXX) -x assembler-with-cpp -c -o $@ $(CXXFLAGS) $<

clean:
	@echo Cleaning:
	$(RM) -rf $(OBJ_DIR)/*

clean_apps:
	@echo Cleaning apps
	$(RM) -rf $(OBJ_DIR)/apps/*

simulate: $(TARGET).elf
	$(FVP_CORSTONE_DIR)/FVP_Corstone_SSE-300_Ethos-U55 -a $< -f config.txt
