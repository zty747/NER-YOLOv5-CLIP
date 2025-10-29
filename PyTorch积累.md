# Swin Transformer

多用堆栈调试
## 1. `flatten` 的作用
**PyTorch 语法：**
```
x.flatten(start_dim, end_dim)
```
- 作用：把输入张量 `x` 在 `start_dim` 到 `end_dim` 之间的维度全部“压平”（合并成一个维度）。
- 常用于把多维张量转换为更低维，方便后续处理。
- python里的flatten(dim)表示，从第dim个维度开始展开，将后面的维度转化为一维.也就是说，只保留dim之前的维度，其他维度的数据全都挤在dim这一维。来自CSDN，如果x.flatten(0)就是`[B*C*H*W]`
**在你的代码里：**
```
x = x.flatten(2)
```
- 假设 `x.shape` 是 `[B, C, H, W]`
- `x.flatten(2)` 等价于 `x.flatten(start_dim=2)`，即把第2和第3维（H和W）合并成一个维度。
- 操作结果：`[B, C, H, W]` → `[B, C, H*W]`
---
## 2. `transpose` 的作用
**PyTorch 语法：**
```
x.transpose(dim0, dim1)
```
- 作用：交换张量的 `dim0` 和 `dim1` 两个维度。
- 常用于调整数据的排列顺序，比如把通道维和序列维互换。
**在你的代码里：**
```
x = x.transpose(1, 2)
```
- 假设此时 `x.shape` 是 `[B, C, HW]`
- `x.transpose(1, 2)` 就是把第1维（C）和第2维（HW）交换
- 操作结果：`[B, C, HW]` → `[B, HW, C]`
---
## 1. `View` 的作用
主要用于==Tensor维度的重构==，即返回一个==有相同数据但不同维度的Tensor==
```python
import torch
temp = [1,2,3,4,5,6] # temp的类型为list，非Tensor
temp = torch.tensor(temp) # 将temp由list类型转为Tensor类型
print(temp) # torch.Size([6])
print(temp.view(2,3)) # 将temp的维度改为2*3
print(temp.view(2,3,1)) # 将temp的维度改为2*3*1
print(temp.view(2,3,1,1)) # 更多的维度也没有问题，只要保证维度改变前后的元素个数相同就行，即2*3*1*1=6
```
![[Pasted image 20250425204827.png]]

---
```
x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]
```
### 2.1 切片（Slicing）
```
x0 = x[:, 0::2, 0::2, :]
```
#### 解释：
- `0::2` 表示从第0行（或列）开始，每隔2个取一个（步长为2）。
- `x[:, 0::2, 0::2, :]`：取所有batch，**偶数行+偶数列**。
- `x[:, 1::2, 0::2, :]`：取所有batch，**奇数行+偶数列**。
- `x[:, 0::2, 1::2, :]`：取所有batch，**偶数行+奇数列**。
- `x[:, 1::2, 1::2, :]`：取所有batch，**奇数行+奇数列**。
这样，把原图的每2×2个patch，分成了4个分块，每个分块 shape 是 `[B, H/2, W/2, C]`。
==start_index:end_index:步长，对每个维度都要处理==

### 2.2 `torch.cat`
```
x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
```
- `cat` 是“拼接”，常用于合并多个张量。
- `-1` 表示在最后一个维度（即通道维）拼接。
此时每个2×2 patch被合并成一个新patch，特征数变为4倍（原来C，现在4C）。
### 2.3 `view`

#### 代码：

```
x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]
```
- `view` 是张量的重塑（reshape），这里把空间维度 `[H/2, W/2]` 合并为一个维度（长度为 `H/2*W/2`），方便后续 transformer 的序列化处理。





---

## 3. 总结流程

1. **切片**：把输入特征图每2×2为一组，分别提取出四个子patch。
2. **拼接**：把这四个子patch在通道维度拼接，得到一个大patch，通道数增加4倍。
3. **重塑**：把空间维度合并为一个序列长度，便于后续 transformer 层处理。


# ResNet

卷积inc与outc是实现==维度变化==，前者限制后者变化。而kernal_size和stride是实现==patch_len/pixel_len/seq_len的变化==（主要是==stride==影响，公式为H2=((H1-K+2P)/S)+1，宽也同理，如卷积核为1X1，步长为1则没有变化,卷积核为3X3，步长为2，padding为1则为原来一半，torch向下取整。卷积核为3X3，步长为1，padding为1则不变），由swin中==L=H×W==启发。对应到线性层似乎没有这种针对seq_len长度变化的操做（linear是对某个patch/pixel/token的维度变化），可能只有池化也能改变长度
relu要求fx的维度与len要和x相同（称H×W为len）
![[Pasted image 20250430131113.png]]

downsampling在maxpool
spatial downsampling with channel doubling在resnet34layer2/3/4的第一个block的第一个conv
==注意mlp是维度减少而不是数据减少，广义的下采样，我更愿意称之为bottleneck==


升级版restnext 分组卷积

# LeNet

学习其用random生成tensor的debug方法，就不用真的图片来看shape了


# ConvNext
问:在convnext网络中，用linear代替k=1，s=1卷积，直接用代码 self.pwconv1 = nn.Linear(dim, 4 * dim) x = self.pwconv1(x)为什么linear不需要展平呢以及linear怎么知道要作用在C维度捏

对于 ==怎么知道要作用在C维度==
linear默认作用在最后一个维度
x = x.permute(0, 2, 3, 1) # [N, C, H, W] -> [N, H, W, C]  
x = self.norm(x)  
x = self.pwconv1(x)
所以用permute把C换到最后一个维度

至于为什么要展平？
linear确实只能作用在1维数据（直线）中，但是C本来就是直线数据
这里联想到vit中的展平，linear是作用在H,W上，patch为2维数据(平面)，所以要展平

只改C可以用conv和linear，conv为B,C,H,W转为B,X,H,W。linear为B,H,W,C转为B,H,W,X，
其中conv要保持图像不变还得保证p=k/2，s=1，当k不等于1，还能捕获平面与交互通道信息
当k=1时就等价于线性只能捕获通道信息了（如cbam的空间注意力为b,2,h,w），k=1与linear能使得两层通道交互，k=7还能和平面交互，但是这是多层通道平面才能使得通道交互，如果是se的条状linear变化什么都交互不了
## 环境变量
为什么要同时新增JAVA_HOME与Path变量？直接设置path为C:\Program Files\Java\jdk-21.0.2不行吗，为什么要通过%JAVA_HOME%\bin来实现一个类似指针的效果呢 如果我把JAVA_HOME与Path变量都设置为C:\Program Files\Java\jdk-21.0.2时，与那么当输入JAVA时，系统会按什么顺序找
你的问题非常好，涉及到了环境变量配置的**规范性与灵活性**。下面详细回答你的每个疑问。

---

## 1. 为什么要同时设置 `JAVA_HOME` 和 `Path` 变量？

**核心原因**：

- ==`JAVA_HOME` 主要是给**第三方软件**或**脚本**用的，不是直接用来查找可执行文件的。==
- `Path` 是操作系统用来查找可执行程序（如 `java`、`javac`）的路径。

### 详细说明：

- `JAVA_HOME`：  
    统一指向JDK安装路径，让Maven、Tomcat、Gradle等工具知道JDK在哪里。  
    很多脚本会用 `%JAVA_HOME%` 变量来拼接出 `bin` 目录（比如 `%JAVA_HOME%\bin`），这样脚本就不怕你JDK换路径，只需改一个地方。
    
- `Path`：  
    操作系统启动程序时会查找这个变量中列出的目录。把 JDK 的 `bin` 目录（即 `%JAVA_HOME%\bin` 或 `C:\Program Files\Java\jdk-21.0.2\bin`）加进去，命令行才能直接用 `java`、`javac` 等命令。
    

### 为什么不直接只用 Path？

你**可以**只在 `Path` 里写 JDK 的 `bin` 路径，这样 `java` 命令也能用。但**很多Java相关工具（如Maven、Tomcat等）要求有 `JAVA_HOME` 环境变量**，否则启动会报错或找不到 JDK。所以两者作用不同，**建议都设置**。

---

## 2. 为什么 `Path` 推荐用 `%JAVA_HOME%\bin` 而不是直接写死路径？

- **灵活性和维护性**：
    
    - 以后如果JDK升级了，比如换成 `C:\Program Files\Java\jdk-22.0.1`，只需改一次 `JAVA_HOME`，`Path` 里自动生效。
    - 如果写死了完整路径，每次升级JDK要改两处甚至多处（比如脚本文件、其他环境变量等），容易出错。
- **像“指针”一样**：
    
    - `%JAVA_HOME%\bin` 实际就是用变量嵌套，类似编程里的指针或引用，便于统一管理。

---

## 3. 如果把 `JAVA_HOME` 和 `Path` 都设置为 `C:\Program Files\Java\jdk-21.0.2`，会怎样？

### 具体情况：

- `JAVA_HOME` 设置为 `C:\Program Files\Java\jdk-21.0.2`，**没问题**，这是通常做法。
    
- 如果 **Path 里直接写 `C:\Program Files\Java\jdk-21.0.2`**（而不是 `C:\Program Files\Java\jdk-21.0.2\bin`），会有问题：
    
    - `Path` 用来查找**可执行文件目录**，JDK 的 `java.exe`、`javac.exe` 等在 `bin` 目录下。
    - 如果你只加了 `C:\Program Files\Java\jdk-21.0.2`，命令行找不到 `java.exe`，因为它不在这个目录里，而是在其 `bin` 子目录。

### 正确做法：

- `JAVA_HOME` = `C:\Program Files\Java\jdk-21.0.2`
- `Path` 里加上 `%JAVA_HOME%\bin`

---

## 4. 命令查找顺序是怎样的？

当你在命令行输入 `java` 时，系统会按照 `Path` 变量里的目录**从前到后**依次查找，找到第一个包含 `java.exe` 的目录就用该文件。

- 如果`Path`里有多个包含`java.exe`的目录，最前面的优先。
- `JAVA_HOME`不直接参与查找命令，只是给其他软件/脚本用的。

---

## 总结

- **`JAVA_HOME`** 主要供软件和脚本引用 JDK 位置用。
- **`Path`** 是操作系统查找命令可执行文件用的，必须指向 `bin` 目录。
- 用 `%JAVA_HOME%\bin` 而不是写死路径，**便于维护和升级**。
- 两者功能不同，一般都要设置。
- `Path` 里必须指向 JDK 的 `bin`，否则命令行找不到 `java` 等工具。

---

### 小贴士

如果你只设置了 `JAVA_HOME`，**命令行是用不了 `java` 命令的**；  
如果你只设置了 `Path`，**部分工具（如Maven）可能会报错找不到 `JAVA_HOME`**。

**规范做法：两者都设，并用 `%JAVA_HOME%\bin` 连接！**

==用户变量优先级大于系统变量==

命令行输入java时，系统是如何在path中的诸多路径找到对应的文件呢，比如我path中增加的是C:\Program Files\Java\jdk-21.0.2\bin，这个路径是如何与java这个关键字对应起来的？，是bin中写好了对应的指令吗
你的问题很有代表性，涉及**命令查找机制**和**操作系统如何识别命令**。下面详细为你解答：
---

## 1. `java` 关键字和 `Path` 变量的关系

当你在命令行窗口输入：
```
java
```
系统会做以下几步：
### 步骤一：查看`Path`变量
- 系统读取当前环境变量`Path`，这是**一组以分号（`;`）分隔的目录路径**。
### 步骤二：依次查找每个目录
- 系统**从前到后**，依次进入`Path`里的每个目录。
- 在每个目录下，查找**有没有名为`java.exe`（Windows）或`java`（Linux/Mac）的可执行文件**。
### 步骤三：找到即执行

- 一旦在某个目录（如`C:\Program Files\Java\jdk-21.0.2\bin`）下找到了`java.exe`，就会**用这个文件来执行命令**。
- 如果在所有目录中都找不到，命令行就会提示“不是内部或外部命令，也不是可运行的程序”。

---

## 2. `java` 关键字和 `bin` 目录的关系

### 为什么是`bin`目录？

- JDK/JRE的`bin`目录下，存放了很多**可执行程序文件**，比如：
    - `java.exe`：Java虚拟机启动器
    - `javac.exe`：Java编译器
    - 还有`javadoc.exe`、`jar.exe`等
### 如何“对应”起来？
- 你在命令行输入`java`，系统本质上是在`Path`定义的各目录下**查找一个“叫 java.exe 的文件”**。
- 当你把 `C:\Program Files\Java\jdk-21.0.2\bin` 加入到 `Path` 变量后，
    - 系统会在这个目录下找到 `java.exe`，
    - 于是`java`命令就被系统“识别”并执行了。
### 不是“关键字”而是文件名
- `java` 不是系统的“关键字”，而是**可执行文件的名字**。
- 操作系统只是把你的输入当成“要找的程序名字”，在各个目录下查找同名文件（带后缀.exe）。
---
## 3. 举个例子

假如你的`Path`里是：
```
C:\CustomTools;C:\Program Files\Java\jdk-21.0.2\bin;C:\Windows\System32
```

- 系统先去`C:\CustomTools`下找有没有`java.exe`，没有就下一个；
- 然后去`C:\Program Files\Java\jdk-21.0.2\bin`找，找到就执行；
- 如果还找不到，再去`C:\Windows\System32`找。
---

## 4. 能否自定义命令？

是的！你可以把自己的可执行文件（或批处理脚本、shell脚本）放进`Path`里的某个目录，比如`D:\MyTools\bin\hello.exe`，然后在命令行直接输入`hello`就能运行。

---

## 5. 总结

- `java`命令能被识别，是因为`Path`变量中包含了JDK的`bin`目录，而该目录下正好有`java.exe`文件。
- 操作系统按顺序查找`Path`中指定的所有目录，找到后就执行。
- “关键字”其实就是**可执行文件名**，不是系统内置的“映射表”或“指令”。

---

如需了解更底层的查找流程（比如Windows的`PATHEXT`、Linux的`which`命令等），或者想尝试自定义命令，欢迎随时提问！

## 6. 通道注意力与空间注意力

SENET
通道注意力
nn.AdaptiveAvgPool2d(1)直接指定输出尺寸（1×1直接写成1） 强制输出固定尺寸，自动调整池化窗口

nn.AvgPool2d 手动指定 kernel_size,stride, padding     逐步下采样，依赖输入尺寸计算输出

显示建模优于可学习得到的注意力推断


SKNET






CBAM
通道注意力+空间注意力
先验与知识点
生成空间注意力特征图中，先要采集两个1,h,w的描述符，再合成一个1,h,w的特征图

采集1,h,w的两个描述符考虑使用pool和k1卷积把c变为1
前者效果好，显示建模优于可学习得到的注意力推断

合成一个1,h,w的特征图考虑用大卷积核与小卷积核
前者效果好，大感受野重要





坐标注意力
AdaptiveAvgPool2d与AvgPool2d，conv一样，默认处理后两维,就是对hw平面处理
nn.AdaptiveAvgPool2d(none,1)是对倒2维保持原状，倒一维变为1
如(B, C, H, W) -->nn.AdaptiveAvgPool2d(none,1) -->(B, C, H, 1)

坐标注意力实际上含有通道注意力的内容，因为是包含通道维度的平面，此外有h方向与w方向所以把空间与通道都建模了（这里的空间是hw平面，和通道不重复）

先验与知识点
cbam的空间注意力把维度先压缩到2后压缩到1导致信息丢失


## 6. ACM




## 6. DNA
E:\python\envs\py38\lib\site-packages\torch\optim\lr_scheduler.py:131: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
  E:\python\envs\py38\lib\site-packages\torch\nn\functional.py:3103: UserWarning: The default behavior for interpolate/upsample with float scale_factor changed in 1.6.0 to align with other frameworks/libraries, and now uses scale_factor directly, instead of relying on the computed output size. If you wish to restore the old behavior, please set recompute_scale_factor=True. See the documentation of nn.Upsample for details. 
  warnings.warn("The default behavior for interpolate/upsample with float scale_factor changed "


---

### 1. mIoU（Mean Intersection over Union，平均交并比）

- **定义**：mIoU 是一种衡量图像分割模型性能的指标。它计算预测分割区域与真实分割区域重叠的程度，具体为两个区域的交集面积除以它们的并集面积。
- **计算方式**：  
    对每个类别计算IoU：
    
    IoU=TPTP+FP+FNIoU = \frac{TP}{TP + FP + FN}IoU=TP+FP+FNTP​
    
    其中，TP（True Positive）是真阳性像素数，FP是假阳性像素数，FN是假阴性像素数。  
    然后对所有类别的IoU取平均值，得到mIoU。
- **用途**：广泛用于语义分割任务，评价模型整体分割效果，数值越大表示模型性能越好。

---

### 2. PD（Probability of Detection，检测概率）

- **定义**：PD 是指在实际存在目标时，模型正确检测到目标的概率，也称为召回率（Recall）。
- **计算方式**：
    
    PD=TPTP+FNPD = \frac{TP}{TP + FN}PD=TP+FNTP​ 见GPT
    
    表示模型检测到的正样本占所有真实正样本的比例。
- **用途**：用于评价模型的敏感度，即漏检率低，能检测到更多真正目标。

---

### 3. FA（False Alarm，误警率）

- **定义**：FA 是指模型错误地将背景或非目标区域识别为目标的概率。
- **常见表达**：
    - 有时用误报率（False Positive Rate, FPR）来表示，计算公式为：
        
        FPR=FPFP+TNFPR = \frac{FP}{FP + TN}FPR=FP+TNFP​
        
    - 也有直接统计错误报警次数的情况。
- **用途**：衡量模型误检的情况，误警率越低说明模型越准确，不容易产生虚假告警。

---

### 总结

|指标|含义|计算公式|作用|
|---|---|---|---|
|mIoU|平均交并比，衡量分割重叠度|TPTP+FP+FN\frac{TP}{TP+FP+FN}TP+FP+FNTP​ 平均|衡量分割模型的整体性能|
|PD|检测概率，召回率|TPTP+FN\frac{TP}{TP+FN}TP+FNTP​|衡量模型的检测能力|
|FA|误警率，误报率|FPFP+TN\frac{FP}{FP+TN}FP+TNFP​|衡量模型误报的频率|



### ROC（Receiver Operating Characteristic，受试者工作特征曲线）

- **定义**：  
    ROC曲线是用于评估二分类模型性能的图形工具。它展示了不同阈值下模型的**真正率（True Positive Rate, TPR）**与**假正率（False Positive Rate, FPR）**之间的权衡关系。
    
- **具体指标**：
    
    - **TPR（真正率，又称召回率 Recall）**：
        
        TPR=TPTP+FNTPR = \frac{TP}{TP + FN}TPR=TP+FNTP​
        
        表示模型正确识别为正类的比例。
        
    - **FPR（假正率）**：
        
        FPR=FPFP+TNFPR = \frac{FP}{FP + TN}FPR=FP+TNFP​
        
        表示模型错误识别为正类的负样本比例。
        
- **绘制方法**：
    
    - 横轴是FPR，纵轴是TPR。
    - 通过调整分类阈值，计算不同阈值下的 (FPR, TPR) 点，连成曲线。
- **意义**：
    
    - ROC曲线越接近左上角（高TPR，低FPR），模型性能越好。
    - 曲线下的面积称为 **AUC (Area Under Curve)**，AUC 值越接近1，模型越优秀。

---

### ROC曲线的应用场景

- 二分类问题，如医学诊断（判断病人是否患病）、欺诈检测等。
- 用于选择最佳阈值，平衡召回率和误报率。
- 比较不同模型的分类性能。

---

### 总结

|指标|含义|计算公式|作用|
|---|---|---|---|
|TPR|真正率（召回率）|TPTP+FN\frac{TP}{TP + FN}TP+FNTP​|衡量模型识别正样本的能力|
|FPR|假正率|FPFP+TN\frac{FP}{FP + TN}FP+TNFP​|衡量模型误识别负样本的概率|
|ROC|TPR vs FPR 曲线|—|评估模型整体分类性能|
|AUC|ROC曲线下面积|—|衡量模型区分正负样本能力|

---

如果你需要，我也可以帮你画一张ROC曲线的示意图，或者给你提供相关代码示例！

==了解图像怎么处理成向量并打包成batch，模型输出的形状，拿到输出计算指标损失函数==

#### 1.图像处理

==这里大体流程：写出load_dataset分割函数，返回train_img_ids,val_img_ids（为列表含各集图片名字）
写TrainSetLoader，接受train_img_ids，在getitem函数完成返回张量对
写Trainer，用train_img_ids实例化TrainSetLoader，并升级为可取batch的DataLoader




==
```python
# 文件: train.py -> class Trainer -> __init__

# Preprocess and load data
input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
```

目的：把许多png格式的image，mask对变为向量并打包为batch，中间还要实现数据增强造更多数据
思路：
1.先搞定处理一个image，mask对为向量的方法，也就是定义__getitem__，再考虑打包
2.数据增强实现的旋转裁剪肯定要在向量化前（向量不太好旋转裁剪），一对image，mask要做同样的处理，不然就对应不上了，旋转裁剪要用工具PIL的Image.open把png变为可处理对象，打开前要获取png的位置与名称，Image.open后的image类型才能旋转裁剪与执行transforms族的方法（ transforms.ToTensor(),transforms.Normalize）
由于mask是黑白图，作者处理为向量用的是np.expand_dims(mask, axis=0).astype('float32')/ 255.0而不是transforms.ToTensor()。

代码：
```python
def __getitem__(self, idx):  
  
img_id = self._items[idx] # idx：('../SIRST', 'Misc_70') 成对出现，因为我的workers设置为了2  
img_path = self.images+'/'+img_id+self.suffix # img_id的数值正好补了self._image_path在上面定义的2个空  
label_path = self.masks +'/'+img_id+self.suffix  
  
img = Image.open(img_path).convert('RGB') ##由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸  
mask = Image.open(label_path)  
  
# synchronized transform  
img, mask = self._sync_transform(img, mask)  
  
# general resize, normalize and toTensor 
input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
if self.transform is not None:  
img = self.transform(img)  
mask = np.expand_dims(mask, axis=0).astype('float32')/ 255.0  
  
  
return img, torch.from_numpy(mask) #img_id[-1]


```



==写好class TrainSetLoader(Dataset)类：==

打碎重建，自底向上，从图片文件夹到张量对，以过程推代码
与ner训练不同，这个是一张张图片，目的应该是拿到tbar = tqdm(self.test_data)来实现for i, ( data, labels) in enumerate(tbar):

1.拿到每张图片的位置，有\dataset\NUDT-SIRST\50_50记录每个文件   拼接出图像与标注的完整路径

==同时完成了分割操作，这提醒我们，分割数据集可以优雅的在train，test文件之外==
```python
def load_dataset (root, dataset, split_method):  
train_txt = root + '/' + dataset + '/' + split_method + '/' + 'train.txt'  
test_txt = root + '/' + dataset + '/' + split_method + '/' + 'test.txt'  
train_img_ids = []  
val_img_ids = []  
with open(train_txt, "r") as f:  
line = f.readline()  
while line:  
train_img_ids.append(line.split('\n')[0])  
line = f.readline()  
f.close()  
with open(test_txt, "r") as f:  
line = f.readline()  
while line:  
val_img_ids.append(line.split('\n')[0])  
line = f.readline()  
f.close()  
return train_img_ids,val_img_ids,test_txt
```

2.获取位置后，就可以进行水平翻转随机裁剪来数据增强并张量化了，这一过程作者选择封装在继承Dataset的TrainSetLoader类的__getitem__里，这样每次取到的都是处理过后的image，mask对了
注意Dataset要一定要实现__getitem__与__len__函数，getitem函数就是定义你给Dataset一个index，Dataset返回给你什么，可以返回多个东西
Dataset就像一个菜谱一样，也不含菜品（张量），但是有原材料（图片）与加工方法（处理函数）。
```python
class TrainSetLoader(Dataset):  
  
  
"""Iceberg Segmentation dataset."""  
NUM_CLASS = 1  
  
def __init__(self, dataset_dir, img_id ,base_size=512,crop_size=480,transform=None,suffix='.png'):  
super(TrainSetLoader, self).__init__()  
  
self.transform = transform  
self._items = img_id  
self.masks = dataset_dir+'/'+'masks'  
self.images = dataset_dir+'/'+'images'  
self.base_size = base_size  
self.crop_size = crop_size  
self.suffix = suffix  
  
def _sync_transform(self, img, mask):  
# random mirror  
if random.random() < 0.5:  
img = img.transpose(Image.FLIP_LEFT_RIGHT)  
mask = mask.transpose(Image.FLIP_LEFT_RIGHT)  
crop_size = self.crop_size  
# random scale (short edge)  
long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))  
w, h = img.size  
if h > w:  
oh = long_size  
ow = int(1.0 * w * long_size / h + 0.5)  
short_size = ow  
else:  
ow = long_size  
oh = int(1.0 * h * long_size / w + 0.5)  
short_size = oh  
img = img.resize((ow, oh), Image.BILINEAR)  
mask = mask.resize((ow, oh), Image.NEAREST)  
# pad crop  
if short_size < crop_size:  
padh = crop_size - oh if oh < crop_size else 0  
padw = crop_size - ow if ow < crop_size else 0  
img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)  
mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)  
# random crop crop_size  
w, h = img.size  
x1 = random.randint(0, w - crop_size)  
y1 = random.randint(0, h - crop_size)  
img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))  
mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))  
# gaussian blur as in PSP  
if random.random() < 0.5:  
img = img.filter(ImageFilter.GaussianBlur(  
radius=random.random()))  
# final transform  
img, mask = np.array(img), np.array(mask, dtype=np.float32)  
return img, mask  
  
def __getitem__(self, idx):  
  
img_id = self._items[idx] # idx：('../SIRST', 'Misc_70') 成对出现，因为我的workers设置为了2  
img_path = self.images+'/'+img_id+self.suffix # img_id的数值正好补了self._image_path在上面定义的2个空  
label_path = self.masks +'/'+img_id+self.suffix  
  
img = Image.open(img_path).convert('RGB') ##由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸  
mask = Image.open(label_path)  
  
# synchronized transform  
img, mask = self._sync_transform(img, mask)  
  
# general resize, normalize and toTensor  
if self.transform is not None:  
img = self.transform(img)  
mask = np.expand_dims(mask, axis=0).astype('float32')/ 255.0  
  
  
return img, torch.from_numpy(mask) #img_id[-1]  
  
def __len__(self):  
return len(self._items)

```






3，接着想要一次取一个batch，就要在创建一个TrainSetLoader类（实质上为Dataset）后，将其作为参数传入torch官方的DataLoader类拿到一个train_data，train_data本身**不是**数据，而是一个**数据的生成器**。它知道如何根据你设定的规则（batch_size, shuffle等）从 `trainset` 中按需生成一批批的数据。Dataset就像一个菜谱一样，也不含菜品（张量），但是有原材料（图片）与加工方法（处理函数）。

再把train_data包装成tbar就可以显示进度条













好的，这是一个非常核心且重要的问题，理解了它就等于理解了 PyTorch 数据处理和训练循环的命脉。我会结合源码的概念，为你详细拆解这个过程。

我们将分四步来讲解：

1. **`Dataset` 类 (以 `TrainSetLoader` 为例): 核心是 `__getitem__`，负责准备“一道菜”**
2. **`DataLoader` 类: 负责当“大堂经理”，将多道菜打包成“一份套餐 (batch)”**
3. **`tqdm` 包装器: 负责给“上菜过程”加上一个“进度条”**
4. **训练循环: 如何“享用”这份带进度条的套餐**

---

### 1. `Dataset` 类：准备单一样本的蓝图

`Dataset` 是一个抽象类，它规定了任何自定义数据集都需要遵守的“协议”或“接口”。在你的代码中，`TrainSetLoader` 就是 `Dataset` 的一个具体实现。

它的核心使命是：**告诉程序如何获取并处理 _单个_ 数据样本**。你可以把它想象成一个厨师的菜谱，`__getitem__` 就是其中最关键的一页，详细描述了如何制作“一道菜”。

#### `__getitem__(self, idx)` 方法详解

这个方法是 `Dataset` 的灵魂。它的输入是一个索引 `idx` (一个整数)，输出是**一个**处理好的数据样本（在你的案例中，是一个图像张量和其对应标签张量的元组 `(image, mask)`)。

让我们一步步分解 `TrainSetLoader` 的 `__getitem__`：

```python
# 文件: utils.py -> class TrainSetLoader
def __getitem__(self, idx):
    # ----------------------------------------------------------------------
    # 步骤 1: 定位原材料
    # DataLoader会给这个方法传递一个整数索引，例如 idx = 42
    # ----------------------------------------------------------------------
    img_id     = self._items[idx]  # 从预先加载的ID列表中，根据索引找到文件名，如 'Misc_70'
    img_path   = self.images+'/'+img_id+self.suffix   # 拼接出完整的图像文件路径
    label_path = self.masks +'/'+img_id+self.suffix   # 拼接出完整的标签文件路径

    # ----------------------------------------------------------------------
    # 步骤 2: 从磁盘加载原材料
    # ----------------------------------------------------------------------
    img = Image.open(img_path).convert('RGB') # 使用Pillow库打开图像，并统一转为RGB格式
    mask = Image.open(label_path)              # 打开对应的标签（mask）

    # ----------------------------------------------------------------------
    # 步骤 3: 烹饪/预处理（数据增强）
    # 这是非常关键的一步，目的是增加数据多样性，防止模型过拟合
    # ----------------------------------------------------------------------
    img, mask = self._sync_transform(img, mask) # 对图像和标签进行一系列随机的、同步的变换
                                                # (如随机翻转、缩放、裁剪)

    # ----------------------------------------------------------------------
    # 步骤 4: 最终加工，转换成PyTorch能理解的格式（张量）
    # ----------------------------------------------------------------------
    if self.transform is not None:
        # self.transform 就是在 train.py 中定义的 transforms.Compose([...])
        # 它包含 ToTensor() 和 Normalize()
        img = self.transform(img)
        # 经过这步，img从一个PIL图像对象，变成了：
        # 1. 一个PyTorch张量
        # 2. 形状从 (H, W, C) 变为 (C, H, W)，例如 (3, 256, 256)
        # 3. 像素值被归一化到 [0, 1] 然后又被标准化

    # 对标签也进行处理
    mask = np.expand_dims(mask, axis=0).astype('float32')/ 255.0
    # 这步将 (H, W) 的标签图，变为 (1, H, W) 的Numpy数组，并归一化

    # ----------------------------------------------------------------------
    # 步骤 5: 上菜！
    # 返回一个元组，包含一个处理好的图像张量和一个标签张量
    # ----------------------------------------------------------------------
    return img, torch.from_numpy(mask)
```

**小结**：`__getitem__` 的作用就像一条流水线，输入一个编号，输出一份标准化的、随时可以送入模型的 `(图像, 标签)` 数据对。

---

### 2. `DataLoader`：高效的批处理引擎

如果说 `Dataset` 是“菜谱”，那么 `DataLoader` 就是一个高效的“大堂经理+后厨调度系统”。它从 `Dataset` 中取数据，然后把它们打包成批次（batch）。

`DataLoader` 是一个 **Python 的可迭代对象 (iterable)**。这意味着你可以用 `for` 循环来遍历它。

让我们来看这行代码：

```python
self.train_data = DataLoader(dataset=trainset, batch_size=4, shuffle=True, num_workers=4, drop_last=True)
```

这行代码创建了一个 `DataLoader` 实例，我们来分析它的参数：

- `dataset=trainset`: 告诉 `DataLoader` 从哪里获取数据。它知道 `trainset` 有 `__getitem__` 和 `__len__` 方法。
- `batch_size=4`: 这是核心参数。`DataLoader` 每次迭代（上菜）时，会打包4个样本。它会调用4次 `trainset.__getitem__(idx)`，得到4个 `(image, mask)` 元组。
- `shuffle=True`: **非常重要**。在每个 epoch 开始时，`DataLoader` 会将所有样本的索引（从 0 到 `len(trainset)-1`）进行一次**随机打乱**。然后按照这个打乱后的顺序去调用 `__getitem__`。这可以防止模型学到数据的顺序，增强泛化能力。
- `num_workers=4`: **性能优化的关键**。这表示 `DataLoader` 会开启4个并行的子进程来预加载数据。当你的主进程（和GPU）正在处理第 N 个批次时，这4个“工人”已经在后台马不停蹄地准备第 N+1, N+2, ... 个批次了。这极大地减少了CPU准备数据的时间，避免GPU“空等”，从而加快整体训练速度。
- `drop_last=True`: 如果你的数据集样本总数不能被 `batch_size` 整除（例如1003个样本，batch_size=4），最后一个批次就只有3个样本。设置 `drop_last=True` 会直接丢弃这个不完整的批次，保证每个批次的尺寸都一样。

#### `train_data` 是什么东西？

`self.train_data` 是一个 `DataLoader` 实例，它是一个**可迭代对象**。它本身**不是**数据，而是一个**数据的生成器**。它知道如何根据你设定的规则（batch_size, shuffle等）从 `trainset` 中按需生成一批批的数据。

当你对它进行迭代时，它内部会执行一个叫做 `collate_fn` (整理函数) 的操作。这个函数负责将 `batch_size` 个单独的样本“整理”成一个批次。

- 它会把4个形状为 `(3, 256, 256)` 的图像张量，在**新的维度0**上堆叠起来，形成一个形状为 `(4, 3, 256, 256)` 的批次图像张量。
- 它会把4个形状为 `(1, 256, 256)` 的标签张量，堆叠成一个形状为 `(4, 1, 256, 256)` 的批次标签张量。

---

### 3. `tqdm`：美观的进度条包装器

`tqdm` 是一个非常流行的库，用于在循环中显示智能、可扩展的进度条。

`tbar = tqdm(self.train_data)`

这行代码的作用是：

1. `tqdm` 接收一个可迭代对象 `self.train_data`。
2. 它返回**另一个新的可迭代对象** `tbar`。

这个新的 `tbar` 对象是一个“代理”或“包装器”。它有和 `self.train_data` 完全相同的迭代行为，但增加了一个额外的功能：**每次你从它这里取走一个元素，它都会更新屏幕上的进度条**。

`tqdm` 会自动调用 `len(self.train_data)` 来获取总的迭代次数（即 `总样本数 / batch_size`），从而可以计算进度百分比和预计剩余时间。

---

### 4. 训练循环：消费数据批次

现在我们来看这部分代码，一切就都说得通了：

```python
# tbar 是一个可迭代对象，所以可以用在 for 循环中
for i, ( data, labels) in enumerate(tbar):
    # ...
```

1. **`for ... in tbar:`**: `for` 循环向可迭代对象 `tbar` 请求下一个元素。
2. `tbar` 内部向 `self.train_data` 请求下一个元素。
3. `self.train_data` 执行我们上面描述的复杂流程：获取4个打乱的索引 -> 让4个worker调用 `__getitem__` -> 收集结果 -> `collate_fn` 堆叠成批次。
4. `self.train_data` 返回一个元组，包含两个批次张量：
    - 一个图像批次张量，形状为 `(4, 3, 256, 256)`。
    - 一个标签批次张量，形状为 `(4, 1, 256, 256)`。
5. `tbar` 接收到这个元组，**更新一下进度条**，然后将元组传递给 `for` 循环。
6. `i, (data, labels)`: `enumerate` 提供了索引 `i`。`(data, labels)` 通过**元组解包**，将元组中的两个张量分别赋值给 `data` 和 `labels` 变量。
    - `data` 现在是形状为 `(4, 3, 256, 256)` 的图像批次张量。
    - `labels` 现在是形状为 `(4, 1, 256, 256)` 的标签批次张量。
7. `data = data.cuda()`: 将这一批次的图像数据从CPU内存转移到GPU显存，准备进行高速计算。
8. `preds = self.model(data)`: 将整批数据送入模型，得到整批的预测结果。
9. `tbar.set_description(...)`: 这是 `tqdm` 对象 `tbar` 提供的另一个便利方法，它允许你在循环中动态地更新进度条前面的描述文字，非常适合用来实时显示损失值。

**总结一下整个流程**：  
`for` 循环驱动一切 -> `tqdm` 包装器负责显示进度 -> `DataLoader` 负责高效地从 `Dataset` 中取数据并打包成批次 -> `Dataset` 负责处理单个数据样本。

这个设计模式将**数据存储**（磁盘上的文件）、**数据表示**（`Dataset`）、**数据迭代**（`DataLoader`）和**模型训练**（循环体）完美地解耦，使得每一部分都可以独立修改和复用，是 PyTorch 框架设计的精髓所在。



##### 虚拟环境安装软件包问题
PyCharm 通常会自动激活项目中的虚拟环境（就是终端操作对象是虚拟环境），但是切换后似乎会出bug，如下图发现终端操作对象是正常环境py3.11
![[Pasted image 20250810110128.png]]

这时候
conda activate tf3 
还是py3.11
似乎是
**路径冲突：** 如果 pip 仍然使用全局环境，确保您的系统 PATH 环境变量没有优先全局 Python。您可以通过运行 `which python` (macOS/Linux) 或 `where python` (Windows) 在终端中检查当前 Python 的路径。



我想为我的虚拟环境安装 pip install torchvision ，但是输入显示我已经安装了torchvision在本地环境中怎么办Requirement already satisfied: torchvision in e:\python\lib\site-packages (0.20.1)

想别的办法，用gui无敌了，在终端那一列可以直接搜索安装


### **1. 步过 (Step Over) - 快捷键 F8**

**作用**：在单步执行时，如果当前行调用了一个函数，"步过"会一次性执行完这个函数，然后停在下一行代码。它不会进入被调用函数的内部。 [[1]](https://blog.csdn.net/qq_42755734/article/details/134520202)[[2]](https://cloud.tencent.com/developer/article/2121405)

**使用场景**：

- 当你确定当前行的函数调用没有问题，或者你暂时不关心该函数的内部实现细节时，使用“步过”可以节省时间，让你专注于当前代码块的逻辑。 [[3]](https://www.jetbrains.com/help/pycharm/stepping-through-the-program.html)
- 如果一个函数内部逻辑很复杂，但你只关心它的返回值是否正确，那么使用“步过”直接获取结果会更高效。

**举例**：

python

运行复制

```python
def calculate_sum(a, b):
    result = a + b
    return result

def main():
    x = 10
    y = 20
    total = calculate_sum(x, y)  # <--- 调试器停留在此行
    print(total)

main()
```

当调试器停在 `total = calculate_sum(x, y)` 这一行时，如果你点击“步过”(F8)，程序会执行 `calculate_sum` 函数，然后直接停在 `print(total)` 这一行。你不会看到 `calculate_sum` 函数内部 `result = a + b` 的执行过程。

---

### **2. 步入 (Step Into) - 快捷键 F7**

**作用**：如果当前行有函数调用，“步入”会进入该函数的内部，从函数的第一行代码开始继续单步执行。 [[1]](https://blog.csdn.net/qq_42755734/article/details/134520202)[[2]](https://cloud.tencent.com/developer/article/2121405) 如果当前行没有函数调用，其效果和“步过”相同。 [[1]](https://blog.csdn.net/qq_42755734/article/details/134520202)

**使用场景**：

- 当你怀疑一个函数可能存在问题，或者想了解其具体的执行流程和内部变量变化时，应该使用“步入”。 [[3]](https://www.jetbrains.com/help/pycharm/stepping-through-the-program.html)[[4]](https://www.jetbrains.com/zh-cn/help/pycharm/stepping-through-the-program.html)
- 在学习和理解一个陌生的函数或库的工作原理时，“步入”是非常有用的工具。

**举例**：  
使用上面的相同代码，当调试器停在 `total = calculate_sum(x, y)` 这一行时，如果你点击“步入”(F7)，调试器会跳转到 `calculate_sum` 函数内部，停在 `result = a + b` 这一行，接下来你可以继续单步执行函数内的代码。

---

### **3. 单步执行我的代码 (Step Into My Code) - 快捷键 Alt+Shift+F7**

**作用**：“单步执行我的代码”是“步入”的一个变种。它只会进入你自己编写的函数（即当前项目中的代码），而会跳过所有Python内置函数、标准库函数或第三方库的内部代码。 [[2]](https://cloud.tencent.com/developer/article/2121405)[[4]](https://www.jetbrains.com/zh-cn/help/pycharm/stepping-through-the-program.html)

**使用场景**：

- 当你只想关注自己业务逻辑的实现，而不希望在调试时进入到例如 `print()`、`len()` 或其他第三方库（如`requests`、`numpy`）的源码中时，这个功能非常实用。 [[4]](https://www.jetbrains.com/zh-cn/help/pycharm/stepping-through-the-program.html)[[5]](https://pycharm.net.cn/stepping-through-the-program.html)
- 这可以让你保持调试的焦点，避免在庞大而复杂的库代码中迷失方向。

**举例**：

python

运行复制

```python
import requests

def get_website_content(url):
    response = requests.get(url)  # <--- 调试器停留在此行
    return response.text

content = get_website_content("http://example.com")
print(content)
```

当调试器停在 `response = requests.get(url)` 这一行时：

- 如果使用“步入”(F7)，你可能会进入 `requests` 库的 `get` 函数源码中。
- 如果使用“单步执行我的代码”(Alt+Shift+F7)，调试器会执行完 `requests.get(url)` 这整行代码，然后停在下一行 `return response.text`，因为它会忽略进入非项目代码的 `requests` 库。 [[2]](https://cloud.tencent.com/developer/article/2121405)

---

### **4. 步出 (Step Out) - 快捷键 Shift+F8**

**作用**：当你已经通过“步入”进入一个函数内部进行调试时，如果剩下的函数代码你不再关心，可以使用“步出”。它会立即执行完当前函数的剩余所有代码，然后返回到该函数被调用的地方的下一行。 [[4]](https://www.jetbrains.com/zh-cn/help/pycharm/stepping-through-the-program.html)[[6]](https://developer.baidu.com/article/details/3231860)

**使用场景**：

- 你通过“步入”进入一个函数，检查了开头的几行代码后，发现问题不在这个函数里，或者已经了解了需要的信息。
- 在一个长循环中，你进入了某次循环的函数调用，检查完毕后想快速结束当前函数的执行，并回到主流程。

**举例**：

python

运行复制

```python
def complex_function():
    # ... 很多行代码 ...
    a = 1  # <--- 你通过“步入”进入函数后，停在这里
    b = 2
    # ... 更多你不想再逐行看的代码 ...
    return a + b

def main():
    result = complex_function()
    print("Done") # <--- 使用“步出”后，会停在这里

main()
```

当你在 `complex_function` 函数内部的 `a = 1` 处时，如果点击“步出”(Shift+F8)，程序会执行完 `complex_function` 剩下的所有代码，然后返回到 `main` 函数中，并停在 `print("Done")` 这一行。

# Kaggle

用手机就很容易注册

流程：
1.上传带数据库的项目文件
2.cd /
3.复制项目路径后  执行cp -r 路径 目的 如：cp -r  /kaggle/input/desnow/Desnowing Desonwing
但是似乎
cp: cannot create directory '/kaggle/input/desnow/Desnowing/pytorch-gradual-warmup-lr': Read-only file system
这是因为不小心cd到/kaggle目录下，要cd到/就是根目录下执行，或者直接cp -r /kaggle/input/warmup22/pytorch-gradual-warmup-lr /Desonwing/pytorch-gradual-warmup-lr

colab可以修改文件内容，kaggle不行,kaggle粘贴mian时要把args = parser.parse_args()改为    args, unknown = parser.parse_known_args() 但是colab自动掉线且上传文件一定要VPN
好想法：在colab跑通后再把文件传kaggle跑，就是colab不好传
命令不行时前面打个！试一下

kaggle改文件只能：删除文件，上传文件，cp文件，但是这样要重启后才能生效
且每次重登后cp与下载的东东就没了
要记得save，似乎save and run就可以离线

![[Pasted image 20250912200940.png]]

如要上传文件，点
![[Pasted image 20250912203352.png]]
upload，把文件压缩后上传，然后cp到需要的目录下
# 深度学习问题

 1.Residual learning 的基本原理？

残差学习的核心思想是引入跳接结构来实现恒等映射。网络不再直接学习复杂的目标输出，而是学习目标输出与输入之间的残差。这种设计有两个关键优势：第一，跨层连接为梯度流动提供捷径，有效缓解深层网络的梯度消失问题；第二，当新增网络层对模型无益时，残差模块能自动退化为恒等映射，从根本上解决网络深度增加导致性能退化的难题。要让最终输出 `relu(out + residual)` 等于 `x`（即 `residual`），最简单的方式就是让主路径的输出 `out` 趋近于0。神经网络通过学习（反向传播），可以将 `conv1` 和 `conv2` 的权重训练得非常小，从而使得 `out` 变成一个接近于零的张量。
”其中conv是 W * x 这样，反向传播中只有W是可学习的参数，residual已经是一个固定的值，所以损失大时（新增网络层对模型无益），只会更新w而不会更新residual，使得w接近0，out + residual接近residual“
**当输入输出维度变化时，为了使残差连接能够进行维度匹配，`residual` 路径上会引入一个包含可学习参数的投影层（通常是1x1卷积）。**（shortcut）
尽管如此，这个投影层的主要目的是为了维度匹配，以实现加法操作，而不是像主路径那样去学习复杂的残差函数。残差学习的核心思想——让主路径学习残差 `F(x)`，并在 `F(x)` 趋近于0时退化为恒等映射——依然成立。即使投影层有参数，它也是为了辅助这个恒等映射的实现。

2.Batch Normailization 的原理，思考 BN、LN、IN 的主要区别。

Batch Normalization的核心操作是在训练过程中对每个特征通道进行动态标准化：在每个批次的数据上实时计算均值和方差，将特征分布调整为均值为0、方差为1的稳定状态，最后通过可训练参数恢复特征表达能力。

三种归一化技术的本质区别在于统计范围：

​​BN​​：基于一个批次内所有样本的同通道数据统计，依赖批量大小

​​LN​​：针对单个样本内部所有==通道==数据统计，与批次无关

IN​​：仅对单样本、单通道的像素数据统计，隔离通道间信息

3.为什么分组卷积可以提升准确率？既然分组卷积可以提升准确率，同时还能降低计算量，分组数量尽量多不⾏吗？

分组卷积通过将通道拆解为独立子组，强制各组学习差异化特征，增强了特征多样性。同时，组间参数隔离使计算量大幅降低。过度分组会割裂通道间的相关性，极端情况下将退化为DW卷积，导致模型丧失跨通道特征融合能力；如果DW加上PW成为深度可分离卷积还能融合 同时，碎片化计算大幅增加内存访问开销，在硬件层面显著降低并行效率。

-------------------------------------------------------------------------------

● 训练HybridSN，然后多测试几次，会发现每次分类的结果都不⼀样，请思考为什么？

主要原因是随机性带来的影响，包括参数初始化、mini-batch 的随机采样顺序、dropout 等正则化操作都会导致不同的收敛路径。如果没有固定随机种子或开启确定性计算，每次训练的最终分类结果就会有所差别，这是深度学习模型的常见现象。

● 如果想要进⼀步提升⾼光谱图像的分类性能，可以如何改进？

在模型上可以引入 Transformer 或注意力机制来更好建模光谱与空间的依赖关系，也可以设计多尺度卷积或残差结构来增强特征提取能力；在数据上可以做光谱维的特征选择与增强，结合半监督学习或迁移学习方法提升模型的泛化性能。 CNN加TR 胡帅学长的模块

● depth-wise conv 和 分组卷积有什么区别与联系？

depth-wise 卷积是将输入特征图的每个通道分别用一个卷积核单独做卷积，不发生跨通道的信息融合；分组卷积是将通道分为若干组，每组内部做标准卷积，比普通卷积参数量更少。depth-wise 卷积可以视为分组卷积的一种特例，即组数等于输入通道数。

● SENet 的注意⼒是不是可以加在空间位置上？

可以扩展到空间维度上，通过学习空间位置的重要性来突出关键区域。

● 在 ShuffleNet 中，通道的 shuffle 如何用代码实现？

先将通道 reshape 成 (batch, groups, channels_per_group, height, width)，再在 groups 和 channels_per_group 这两个维度上做转置，最后再 reshape 回 (batch, channels, height, width)。

 
# Non-Local

==也许上下采样就是单纯扩大缩小==
==可以有：通道下采样2倍  长宽上采样2倍==
### 1. `contiguous()` 的作用

在 PyTorch 中，张量（Tensor）的数据在内存中是按一定顺序存储的。当您执行 `view()`、`permute()`、`transpose()` 等操作时，PyTorch 往往不会立即复制数据，而是返回一个“视图”（view）或一个带有不同步幅（stride）的新张量。这意味着新张量仍然指向原始张量的数据，只是改变了访问数据的方式。

- **非连续（Non-contiguous）张量**: 当一个张量的数据在内存中不是按照其逻辑顺序（例如，行优先或列优先）连续存储时，它就是非连续的。例如，对一个行优先存储的张量进行 `transpose()` 操作后，它的列现在变成了行，但数据在内存中的物理存储顺序并没有改变，所以它就变成了非连续的。
    
- **`contiguous()` 方法**: `contiguous()` 方法的作用是返回一个内存连续的张量。如果原张量已经是连续的，则返回自身；如果是非连续的，则会创建一个新的张量，并将原张量的数据复制到新张量中，使其在内存中是连续存储的。
    

**为什么需要 `contiguous()`？**

某些 PyTorch 操作（尤其是那些需要直接访问底层数据块的操作，如 `view()`、某些 C++ 扩展或 CUDA 内核）要求输入张量是内存连续的。当你对一个张量执行 `permute()` 或 `transpose()` 之后，通常会得到一个非连续的张量。如果紧接着要进行 `view()` 操作，并且 `view()` 的新形状与当前非连续的内存布局不兼容，PyTorch 就会报错。在这种情况下，你需要先调用 `contiguous()` 来确保数据在内存中是连续的，这样 `view()` 才能正确地重新解释数据。


### 2. 高维矩阵乘法 `torch.matmul(x_theta, x_phi)`

我们有以下两个张量：

- `x_theta` 的形状： `(4, 4096, 512)`
- `x_phi` 的形状： `(4, 512, 4096)`
==注意（4096, 512）只是一个长方形，（4096, 512）@（512，4096）是学过的==

`torch.matmul` 支持“广播”（broadcasting）到批处理维度。当两个张量都有批处理维度（即维度 > 2）时，`torch.matmul` 会将批处理维度对齐，并对每个批次独立执行矩阵乘法。

**具体规则如下：**

1. **批处理维度（Batch Dimensions）**: `torch.matmul` 会从右往左比较除了最后两个维度之外的所有维度。这些维度必须匹配，或者其中一个为 1（可以广播），或者其中一个不存在（也可以广播）。在你的例子中，两个张量都有一个批处理维度 `4`，它们是匹配的。所以，最终结果的批处理维度也是 `4`。
    
2. **矩阵乘法维度（Matrix Dimensions）**: 对于每个批次，`torch.matmul` 会执行标准的矩阵乘法。
    
    - `x_theta` 的最后两个维度是 `(4096, 512)`。
    - `x_phi` 的最后两个维度是 `(512, 4096)`。
    
    矩阵乘法的规则是 `(A, B) @ (B, C) = (A, C)`。  
    在这里，`A = 4096`，`B = 512`，`C = 4096`。
    
    所以，对于每个批次，`torch.matmul` 会计算 `(4096, 512)` 乘以 `(512, 4096)`，结果是 `(4096, 4096)`。
    

**最终结果的形状：**

结合批处理维度和矩阵乘法维度，`mul_theta_phi = torch.matmul(x_theta, x_phi)` 得到的张量形状将是：

`(4, 4096, 4096)`

# HybirdSN


![[Pasted image 20250901153946.png]]

nonlocal
![[Pasted image 20250901155231.png]]



## 视频3D卷积

好的，我们来举一个PyTorch中3D卷积处理视频的例子。

在PyTorch中，3D卷积层的输入张量通常期望的形状是 `(N, C_in, D, H, W)`，其中：

- `N` 是批次大小 (Batch size)
- `C_in` 是输入通道数 (Input channels)，对于彩色视频，这通常是3 (RGB)
- `D` 是深度 (Depth)，对于视频，这代表帧数 (Number of frames)
- `H` 是高度 (Height)
- `W` 是宽度 (Width)

卷积核的形状通常是 `(C_out, C_in / groups, kD, kH, kW)`，其中 `kD, kH, kW` 分别是卷积核的深度、高度和宽度。

下面是一个具体的例子：

```python
import torch
import torch.nn as nn

# 1. 定义输入视频数据
# 假设我们有一个批次（batch）中有2个视频，每个视频是彩色（3通道），
# 有16帧，每帧是64x64像素。
# 形状: (Batch_size, Channels, Depth, Height, Width)
batch_size = 2
in_channels = 3  # RGB
num_frames = 16  # 视频帧数 (Depth)
height = 64
width = 64

# 创建一个随机的输入视频张量
input_video = torch.randn(batch_size, in_channels, num_frames, height, width)
print(f"输入视频的形状: {input_video.shape}")

# 2. 定义3D卷积层
# 参数解释:
#   in_channels: 输入特征图的通道数 (这里是3，因为是RGB视频)
#   out_channels: 输出特征图的通道数 (你可以自定义，例如16个不同的特征)
#   kernel_size: 卷积核的大小。可以是一个整数 (所有维度相同)，也可以是一个元组 (kD, kH, kW)。
#                这里我们用 (3, 3, 3) 表示深度、高度、宽度都是3的卷积核。
#                这意味着卷积核会跨越3帧，并在每帧的3x3区域上进行卷积。
#   stride: 卷积核的步长。可以是一个整数，也可以是一个元组 (sD, sH, sW)。
#           这里我们用 (1, 1, 1) 表示在所有维度上步长都是1。
#   padding: 填充。可以是一个整数，也可以是一个元组 (pD, pH, pW)。
#            这里我们用 (1, 1, 1) 表示在所有维度上都进行1像素的填充。
#            填充的目的是为了保持输出特征图的尺寸。
#            对于 kernel_size=3, padding=1, stride=1 的情况，输出尺寸通常与输入尺寸相同。
conv3d_layer = nn.Conv3d(
    in_channels=in_channels,
    out_channels=16,          # 输出16个特征图
    kernel_size=(3, 3, 3),    # 卷积核大小 (深度, 高度, 宽度)
    stride=(1, 1, 1),         # 步长 (深度, 高度, 宽度)
    padding=(1, 1, 1)         # 填充 (深度, 高度, 宽度)
)

print(f"\n定义的3D卷积层: {conv3d_layer}")
print(f"3D卷积层的权重形状: {conv3d_layer.weight.shape}") # (out_channels, in_channels/groups, kD, kH, kW)
print(f"3D卷积层的偏置形状: {conv3d_layer.bias.shape}")

# 3. 执行前向传播
output_feature_map = conv3d_layer(input_video)

# 4. 打印输出特征图的形状
print(f"\n输出特征图的形状: {output_feature_map.shape}")

# 5. 验证输出形状的计算
# 输出维度 D_out = floor((D_in - kD + 2*pD) / sD) + 1
# 输出维度 H_out = floor((H_in - kH + 2*pH) / sH) + 1
# 输出维度 W_out = floor((W_in - kW + 2*pW) / sW) + 1

# 对于本例:
# D_in = 16, kD = 3, pD = 1, sD = 1  => D_out = floor((16 - 3 + 2*1) / 1) + 1 = floor(15 / 1) + 1 = 15 + 1 = 16
# H_in = 64, kH = 3, pH = 1, sH = 1  => H_out = floor((64 - 3 + 2*1) / 1) + 1 = floor(63 / 1) + 1 = 63 + 1 = 64
# W_in = 64, kW = 3, pW = 1, sW = 1  => W_out = floor((64 - 3 + 2*1) / 1) + 1 = floor(63 / 1) + 1 = 63 + 1 = 64

expected_output_depth = (num_frames - 3 + 2*1) // 1 + 1
expected_output_height = (height - 3 + 2*1) // 1 + 1
expected_output_width = (width - 3 + 2*1) // 1 + 1

print(f"\n预期输出深度: {expected_output_depth}")
print(f"预期输出高度: {expected_output_height}")
print(f"预期输出宽度: {expected_output_width}")

# 验证输出通道数是否等于 out_channels
print(f"输出通道数是否等于 {conv3d_layer.out_channels}: {output_feature_map.shape[1] == conv3d_layer.out_channels}")

```

**代码解释：**

1. **`input_video` 定义：** 我们创建了一个模拟的视频张量。`torch.randn` 用于生成随机数据，形状为 `(2, 3, 16, 64, 64)`，表示2个视频，每个视频3个颜色通道，16帧，每帧64x64像素。
2. **`nn.Conv3d` 定义：**
    - `in_channels=3`: 因为输入视频是RGB，所以输入通道是3。
    - `out_channels=16`: 这个卷积层将学习提取16种不同的特征，所以输出会有16个通道。
    - `kernel_size=(3, 3, 3)`: 卷积核在深度（帧数）、高度和宽度上都是3。这意味着每个卷积操作会考虑3个连续帧的数据，以及每帧中3x3的像素区域。
    - `stride=(1, 1, 1)`: 卷积核在所有三个维度上都以1步长移动。
    - `padding=(1, 1, 1)`: 在所有三个维度上都添加1圈填充。对于 `kernel_size=3` 和 `stride=1`，`padding=1` 可以保证输出的深度、高度和宽度与输入相同。
3. **前向传播：** `conv3d_layer(input_video)` 执行卷积操作。
4. **输出形状：** 最终输出的 `output_feature_map` 形状为 `(2, 16, 16, 64, 64)`。
    - `2` 是批次大小，保持不变。
    - `16` 是 `out_channels`，表示输出了16个特征图。
    - `16, 64, 64` 是经过卷积操作后，深度、高度和宽度的尺寸。在这个例子中，由于 `kernel_size=3`, `stride=1`, `padding=1`，所以这三个维度的大小与输入保持一致。

这个例子展示了如何使用PyTorch的`nn.Conv3d`来处理视频数据，以及输入输出形状的对应关系。
![[Pasted image 20250902153305.png]]- **丢失颜色语义信息（核心问题）：** 这是最大的问题。通过 `reshape`，原始的RGB颜色通道信息被“展平”到了深度维度。卷积核不再知道它正在处理的是红色、绿色还是蓝色通道的数据。它只是将48个“切片”视为一个连续的序列。这使得模型无法区分颜色变化和时间变化，也无法利用颜色通道之间的固有关系。
- **时空特征学习效率低下：**
    - **混淆时间与通道：** 卷积核在深度维度上滑动时，可能会从一个颜色通道的末尾滑到下一个颜色通道的开头（例如，从R通道的最后一帧到G通道的第一帧）。这种跨越语义边界的滑动会使得学习到的特征变得混乱和不具意义。
    - **难以捕获有意义的运动：** 视频中的运动是发生在特定颜色背景下的。这种 `reshape` 破坏了这种背景，使得模型难以有效地学习到物体在空间和时间上的连续运动模式。

==让帧数做D的好处（这里D代表与3D卷积核做卷积的维度，联想2D卷积，与2D卷积核做卷积的部分能捕获这个感受野内的特征）==
- **保留语义信息：** 这是最主要的优势。3D卷积层能够明确地识别并处理3个独立的颜色通道（RGB）。卷积核在每个通道上学习特征，然后将这些通道的特征进行融合。这意味着模型能够理解颜色信息的变化，以及这些变化如何与空间和时间上的运动相关联。
- **有效捕获时空特征：** 3D卷积的设计初衷就是为了同时捕获空间（H, W）和时间（D）上的特征。`in_channels=3` 允许卷积核在保持颜色通道语义独立性的同时，学习跨帧的运动模式和空间纹理。例如，它可以学习到红色物体在视频中的移动轨迹，或者绿色背景的变化。
- **符合数据物理结构：** 视频数据天然就是多通道（颜色）和多帧（时间）的组合。这种处理方式与数据的实际物理生成过程和语义结构高度吻合。
- **模型可解释性强：** 学习到的特征更具可解释性，因为它们是在明确的颜色通道背景下提取的
## 高光谱3D卷积

==根据数据的物理结构和任务目标，3D 卷积的“深度”维度被赋予了不同的语义，以最大化模型从数据中提取有用特征的能力。==

### 视频中的 3D 卷积

- **输入数据形状 (PyTorch 习惯):** `(N, C_in, D, H, W)`
    
    - `N`: 批次大小 (Batch Size)
    - `C_in`: 输入通道数 (Input Channels)，例如 RGB 视频的 3
    - `D`: 深度维度，在这里是**时间帧数 (Frames)**
    - `H`: 高度 (Height)
    - `W`: 宽度 (Width)
 “通常把通道维度（rgb视频为3）作为C_in，视频帧数作为conv的D”
    
- **原因：**
    
    1. **捕获时间动态 (Temporal Dynamics):** 视频的核心在于其时间序列上的变化。3D 卷积的主要目的是在空间维度 (H, W) 的基础上，进一步捕获**时间维度 (D)** 上的特征。例如，它可以学习到物体的运动轨迹、动作的演变等。
    2. **通道的独立性 (Channel Independence):** 视频的 RGB 通道是相互关联但又相对独立的颜色信息。在每个时间帧内，RGB 值共同构成了图像的颜色。==3D 卷积通常会对每个输入通道独立应用一组 3D 滤波器，然后将结果相加。这意味着，卷积核在空间和时间上滑动时，它会同时考虑红、绿、蓝三个通道的信息，但这些通道本身是作为独立的输入特征图来处理的。==你可以理解为，对于每个时间步的每个像素，它有 3 个值（R, G, B），这 3 个值是这个像素的“特征”，而不是这个像素的“深度”。

### 高光谱图像中的 3D 卷积

- **输入数据形状 (PyTorch 习惯):** `(N, C_in, D, H, W)`
    - `N`: 批次大小
    - `C_in`: 输入通道数，通常为 1（因为光谱波段本身就是深度）
    - `D`: 深度维度，在这里是**光谱波段数 (Spectral Bands)**
    - `H`: 高度 (Patch Height)
    - `W`: 宽度 (Patch Width)
 “在高光谱图像的3D卷积中，把光谱维度（通道维度）作为conv的D”
- **原因：**
    
    1. **捕获光谱特征 (Spectral Features) 和空间-光谱联合特征：** 高光谱图像的核心在于每个像素点的完整光谱曲线，这条曲线包含了物质的独特“光谱指纹”。3D 卷积在这里的目的是在空间维度 (H, W) 的基础上，进一步捕获**光谱维度 (D)** 上的特征。它允许卷积核同时在空间邻域内和光谱波段之间滑动，从而学习到：
        - **光谱相关性：** 识别不同波段之间反射率的变化模式，这对于区分不同地物至关重要。
        - **空间-光谱联合特征：** 学习到某个空间模式（例如边缘、纹理）在不同光谱波段下的表现，或者某个光谱特征在空间上的分布。这比单独处理空间或光谱特征更强大。
    2. **“通道”即“深度”：** 在高光谱图像中，200 个波段（或 PCA 后的 30 个主成分）**就是**数据的“深度”或“维度”。它们不是像 RGB 那样是同一像素在不同颜色空间上的三个独立测量，而是同一像素在不同波长上的连续测量。因此，将它们作为 3D 卷积的深度维度 `D` 是最自然和有效的方式，使得卷积核可以直接在光谱维度上学习特征。`C_in` 之所以是 1，是因为每个像素在每个空间位置上只有一个“光谱值”，这些光谱值在不同的波段上构成了第三个维度。

### 总结核心区别：

根本原因在于**第三个维度所代表的物理含义和我们希望模型从中学习到的信息类型不同**：

- **视频：** 第三个维度是**时间**。我们希望模型学习**时间动态和运动信息**。RGB 通道是每个时间步的独立颜色信息。
- **高光谱图像：** 第三个维度是**光谱波段**。我们希望模型学习**光谱特征和空间-光谱的联合特征**。光谱波段本身就是数据的“深度”或“维度”。
==对于conv3d的init，有conv3d(in_channel,out_channel,kernel_size,stride,pading),
对于conv3d的forward，有conv1(batch_size,in_channel,D,H,W),
所以输入向量要处理成(batch_size,in_channel,D,H,W)的形式
与conv2类似，in_channel维度的数据会被独立应用一组 3D 滤波器，然后将结果相加，会同时考虑in_channel每个维度的信息。同时可以使得in_channel维度的数据得到交互（se，ecanet）
关键词：同时考虑
D,H,W维度的数据会经过卷积学习特征
关键词：捕获特征
如(2,3,16,64,64) 经过 conv3d(3,4,(3,3,3),(1,1,1),(1,1,1))

==
# DAnet
![[Pasted image 20250901160637.png]]





素材

swin transformer的窗口注意力 与 ssgformer的group注意力

前者是patch级别的，4X4的patch，通过==卷积==(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)来形成embedding（vit也是这样）（此时一个patch形成一个整体，与pixel或token地位相同，有着自己的embedding，正如pixel的channel）

后者


SVD：https://www.bilibili.com/video/BV1ExWxesEVf/?spm_id_from=333.337.search-card.all.click&vd_source=b3576a15f62f5b81baddc038bffff4fb
 25分半
问题：kc是不是没有用到 用到了

### Soble


索伯算子主要用于边界检测，原理是首先用两个卷积因子分别对像素点做卷积，从而可得出该点在这两个方向上的亮度差分近似值。将xy两个方向的亮度近似值进行结合之后，即可得到该点亮度。 得到的是一个数，如果这个值大于了阈值，那么这个点就可以被考虑成边缘点。
通过计算图像像素强度的梯度来识别边缘。它使用两个3×33 \times 33×3的卷积核，分别检测水平和垂直方向的亮度变化，适用于提取图像中物体与背景的分界线


![[Pasted image 20250905112302.png]]


参考资料：https://blog.csdn.net/2501_90215210/article/details/146469777
### SVD

SVD全名奇异值分解，通常用于图片压缩。原理是任何一种线性变换都能用旋转-压缩-旋转来表示，而矩阵就是一种线性变化，所以任何矩阵都能分解为旋转矩阵-压缩矩阵-旋转矩阵。
![[Pasted image 20250905101022.png]]
而压缩矩阵是由奇异值从左上到右下递减的一个对角矩阵，所以原矩阵A可以表示成多个秩1矩阵相加。而1秩矩阵可以表示成两个向量相乘，节约储存空间，所以可以用来压缩矩阵。且由越大奇异值构成的秩1矩阵越接近原矩阵，就像泰勒展开式一样，我们取前越多项相加就会越接近原矩阵（取所有项就等于原矩阵，但是储存空间也等于原图【甚至大于，因为多了奇异值】）
![[Pasted image 20250905101117.png]]


![[Pasted image 20250905101311.png]]
![[Pasted image 20250905111857.png]]

参考资料：https://www.bilibili.com/video/BV1MT4y1y75x
所以SVD通常用于图像压缩，但在论文中，作者提出：传统边缘检测(sobel)侧重捕捉高频边缘特征，而奇异值分解（SVD）则在低频域分析退化纹理。（ Conventional edge detection captures high-frequency edge features, while Singular Value Decomposition (SVD) analyzes degradation texturesin the low-frequency domain. ）
后面也重复提到One is the Sobel operator which detects high frequency information by highlighting intensity changes. The other,Singular Value Decomposition (SVD), finds low frequency knowledge of the degraded image. 











![[Pasted image 20250905112356.png]]


The features obtained from the two filter operations, FSobel and FSV D, both in H×W×1 , ==capture different information from the same input ID==. H and W denote the spatial dimensions at current
stage. ==This complementary spectral analysis enables early capture of various types of degradation, which can provide valuable spectral features for effective restoration.==
作者提到这两种==互补==的手段可以捕捉早期退化









## HCANet
### squeeze

`torch.squeeze()` 用于移除张量中所有或指定维度中大小为 1 的维度。
**功能：**

- 如果未指定 `dim` 参数，`torch.squeeze()` 会移除输入张量中所有大小为 1 的维度。
- 如果指定了 `dim` 参数，`torch.squeeze()` 只会在该指定维度的大小为 1 时才移除它。如果指定维度的大小不为 1，则张量形状不变。

**语法：**  
`torch.squeeze(input, dim=None, *, out=None)` 

**参数：**

- `input` (Tensor)：要操作的输入张量。
- `dim` (int, optional)：可选参数，指定要移除大小为 1 的维度索引。可以是正数或负数索引。

**返回值：**

- 一个新张量，其维度中所有大小为 1 的维度（或指定维度）已被移除。 
- 返回的张量与输入张量共享底层数据存储，这意味着对其中一个的修改会影响另一个。

```python
import torch

# 示例 1: 未指定 dim 参数
x = torch.randn(1, 3, 1, 4, 1)
print("原始张量 x 的形状:", x.shape)
# 输出: 原始张量 x 的形状: torch.Size([1, 3, 1, 4, 1])

y = torch.squeeze(x)
print("squeeze 后张量 y 的形状:", y.shape)
# 输出: squeeze 后张量 y 的形状: torch.Size([3, 4])

# 示例 2: 指定 dim 参数
x = torch.randn(1, 3, 1, 4, 1)
print("\n原始张量 x 的形状:", x.shape)

# 移除索引为 0 的维度 (大小为 1)
y_dim0 = torch.squeeze(x, dim=0)
print("squeeze(dim=0) 后张量 y_dim0 的形状:", y_dim0.shape)
# 输出: squeeze(dim=0) 后张量 y_dim0 的形状: torch.Size([3, 1, 4, 1])

# 尝试移除索引为 1 的维度 (大小为 3, 不为 1)
y_dim1 = torch.squeeze(x, dim=1)
print("squeeze(dim=1) 后张量 y_dim1 的形状:", y_dim1.shape)
# 输出: squeeze(dim=1) 后张量 y_dim1 的形状: torch.Size([1, 3, 1, 4, 1]) (形状不变)

```
### unsqueeze
`torch.unsqueeze()` 用于在张量的指定位置插入一个大小为 1 的新维度。
```python
import torch

# 示例 1: 1D 张量添加维度
x = torch.tensor([1, 2, 3, 4])
print("原始张量 x 的形状:", x.shape)
# 输出: 原始张量 x 的形状: torch.Size([4])

# 在索引 0 处添加维度
y_dim0 = torch.unsqueeze(x, 0)
print("unsqueeze(dim=0) 后张量 y_dim0 的形状:", y_dim0.shape)
# 输出: unsqueeze(dim=0) 后张量 y_dim0 的形状: torch.Size([1, 4])
print("y_dim0:\n", y_dim0)
# 输出: y_dim0:
#  tensor([[1, 2, 3, 4]])

# 在索引 1 处添加维度
y_dim1 = torch.unsqueeze(x, 1)
print("unsqueeze(dim=1) 后张量 y_dim1 的形状:", y_dim1.shape)
# 输出: unsqueeze(dim=1) 后张量 y_dim1 的形状: torch.Size([4, 1])
print("y_dim1:\n", y_dim1)
# 输出: y_dim1:
#  tensor([[1],
#          [2],
#          [3],
#          [4]])

# 示例 2: 2D 张量添加维度 (常用于添加批次维度或通道维度)
image = torch.randn(3, 256, 256) # 假设这是一个彩色图像 (C, H, W)
print("\n原始图像张量 image 的形状:", image.shape)
# 输出: 原始图像张量 image 的形状: torch.Size([3, 256, 256])

# 添加批次维度 (通常在最前面)
batch_image = torch.unsqueeze(image, 0)
print("添加批次维度后 batch_image 的形状:", batch_image.shape)
# 输出: 添加批次维度后 batch_image 的形状: torch.Size([1, 3, 256, 256])

# 假设要添加一个额外的通道维度 (例如，在某些模型中需要 (H, W, C) -> (H, W, 1, C))
# 在索引 2 处添加维度
expanded_image = torch.unsqueeze(image, 2)
print("在索引 2 处添加维度后 expanded_image 的形状:", expanded_image.shape)
# 输出: 在索引 2 处添加维度后 expanded_image 的形状: torch.Size([3, 256, 1, 256])
```

### 思想收获：
很好的结合了局部性，全局性，多尺度性


## OKNet

### FFT
|   |   |
|---|---|
|[`fft2`](https://docs.pytorch.org/docs/1.11/generated/torch.fft.fft2.html#torch.fft.fft2 "torch.fft.fft2")|Computes the 2 dimensional discrete Fourier transform of `input`.|
|[`ifft2`](https://docs.pytorch.org/docs/1.11/generated/torch.fft.ifft2.html#torch.fft.ifft2 "torch.fft.ifft2")|Computes the 2 dimensional inverse discrete Fourier transform of `input`.|
![[Pasted image 20250910170242.png]]

![[Pasted image 20250910170426.png]]
好思路，在频域计算损失函数
![[Pasted image 20250914005018.png]]
#### FFT理解到下面就好了
![[88191e15fb6881a40736f50e8aefab4c.jpg]]

![[Cache_-3fe4ffc79d9a4626..jpg]]
已经知道时域信号是如何分解为频域谱和相位谱了，不用知道分解的数学推倒  
图像也是同理，不要知道推倒过程，只要知道他能分解成很多条纹，得到频谱与相位（相位类似上面时域理解）

![[Pasted image 20250913204332.png]]
上图可见图片分解可以得到很多条纹图（等价于不同频率（小欧米噶）的三角函数），这些条纹图可以用一个频谱图表示，但是缺少相位信息还是不能还原（这些条纹图的相对位置位置，可以联系三角函数相位），拿到相位图后就可以还原了

![[Pasted image 20250913204303.png]]
问：fft2矩阵内容与频谱图关系？

问：哪来的复数
![[Pasted image 20250913205315.png]]

欧拉公式

---
只要知道fft2的输出结果有：
在未移位的`fft_img`中，低频分量位于图像的四个角（即索引(0,0)附近），而高频分量位于图像中心。这是因为傅里叶变换的默认输出是未中心化的
移位后，实际上就是频谱图，图像中心点（H/2, W/2）代表直流分量（最低频率），从中心向外，频率逐渐增加。这种布局更符合人类视觉习惯，也便于滤波操作。
当我们将`fft_shifted`与`mask`相乘时，中心区域（低频）被保留（乘以1），而外围区域（高频）被抑制（乘以0）。 这相当于在频域中直接去除高频分量，只保留低频分量。

```python
# --- 针对 1/4 尺寸的图像 ---
# 1. 对真实标签图像进行2D FFT
label_fft1 = torch.fft.fft2(label_img4, dim=(-2,-1))
# 2. 将复数结果分离为实部和虚部
label_fft1 = torch.stack((label_fft1.real, label_fft1.imag), -1)

# 3. 对模型预测图像进行2D FFT
pred_fft1 = torch.fft.fft2(pred_img[0], dim=(-2,-1))
# 4. 将复数结果分离为实部和虚部
pred_fft1 = torch.stack((pred_fft1.real, pred_fft1.imag), -1)

# ... (对 1/2 尺寸和原尺寸的图像重复同样的操作) ...

# 5. 在频域上计算损失
f1 = criterion(pred_fft1, label_fft1)
f2 = criterion(pred_fft2, label_fft2)
f3 = criterion(pred_fft3, label_fft3)
loss_fft = f1+f2+f3
```

## AutoDL
### VScode

点到python文件右小角就可以换解释器了
![[Pasted image 20250915160815.png]]
新版本vs无法连接auto的低版本ubantu，所以用低版本vs
低版本vs左下角连接ssh，连接有两种方式，一种是autodl在最上方输入服务器名后自动写入sshconfig，第二种直接在sshconfig写入信息，sshconfig就是记录连接服务器的信息

连接后python扩展要重新下载（vs不像pycharm为py而生，下载python扩展可以更好的支持python，注意不是解释器，这个只是类似使得python语法高亮的东西）

进服务器进入root目录，上传文件似乎可以直接拖拽上传，vs不方便的视乎是调试 
多利用复制文件当前路径来cd 


## Restormer

“channel-wise”是用来修饰“aggregation of local context”的。它的意思是：**在每个通道上（channel-wise），独立地（independently）聚合其局部的空间上下文（local context）**。
channel-wise pooling 也是在每一个通道上pooling，(b,c,hw)->(b,c,1)

门控机制

###  什么是归纳偏置 (Inductive Bias)？

**归纳偏置是机器学习模型在学习过程中，对“什么样的解决方案更好”所做的一组内置的、先验的假设。**

简单来说，为了让模型能从有限的训练数据中学会如何处理无限的未知数据（即“泛化”），我们必须给==模型一些“提示”或“规则”==。这些“提示”就是归纳偏置，它限定了模型的学习范围，引导它朝着我们认为更可能正确的方向去学习。==感觉就是先验知识，如ssg的分组注意力，相近的才要做注意力就是人预先判断的而不是学习的==

**举几个经典的例子：**

- **卷积神经网络 (CNN) 的归纳偏置**：
    
    1. **局部性 (Locality)**：假设图像中的像素与其邻近像素的关系最密切。这个假设通过小的卷积核（如 3x3）来实现，使得网络优先学习局部特征。
    2. **平移不变性/等变性 (Translation Invariance/Equivariance)**：假设图像中的一个物体（比如一只猫）无论出现在左上角还是右下角，它仍然是同一只猫。这个假设通过在整个图像上共享卷积核（权重共享）来实现，使得模型学到的特征检测器可以应用于图像的任何位置。
    
    - 这个偏置非常适合图像任务，所以 CNN 在图像领域非常成功且高效。
- **Transformer 的归纳偏置**：
    
    - 标准的 Vision Transformer 的归纳偏置**非常弱**。它的自注意力机制假设**图像中的任何一个像素块都可以与其他所有像素块直接产生关联**，它没有“局部性”的先验假设。
    - 这使得 Transformer 非常灵活和强大，但也导致它需要海量的数据才能自己“悟出”局部性等图像的基本规律。这也是为什么 Restormer 要特意在注意力模块和前馈网络中加入深度卷积（DW Conv），目的就是**人为地为 Transformer 补充上“局部性”这个强大的归纳偏置**。
- **循环神经网络 (RNN) 的归纳偏置**：
    
    - **序列性/时间依赖性 (Sequentiality)**：假设数据是按顺序排列的，并且当前时刻的状态与之前的状态有关。这个偏置非常适合处理语言、时间序列等顺序数据。

**为什么归-纳偏置很重要？**  
一个好的归纳偏置可以让模型：

- **学习效率更高**：不需要从零开始探索所有可能性。
- **所需数据更少**：因为很多“规则”已经预设好了。
- **泛化能力更强**：更不容易在训练数据上过拟合。

注意力**作用在通道维度而非空间维度**。您担心这会缺少像素级的空间注意力，从而影响效果，这是一个非常合理且关键的疑问。

1.感觉可以加上分组空间/线性注意力

2.dw后加shuffle或PW使得通道交互？

## basicSR

```bash
pip install basicsr
会自动下载需要的包
```

问题：
E:\python\Lib\site-packages\basicsr\data\degradations.py", line 8, in 
No module named 'torchvision.transforms.functional_tensor'
```python
from torchvision.transforms.functional_tensor import rgb_to_grayscale
```
将其修改为：
```python
from torchvision.transforms.functional import rgb_to_grayscale
```

注意E:\AI\deep-learning-for-image-processing-master\BasicSR-master\basicsr\data\degradations.py中已经是from torchvision.transforms.functional import rgb_to_grayscale很奇怪两种写法不一样



文字gbk编码见poe，就是window是gbk，yaml是utf-8
我直接改成utf-8

tf3中报错，似乎numpy在又不在
```
(tf3) PS E:\AI\deep-learning-for-image-processing-master\BasicSR-master> pip install --force-reinstall --no-deps numpy==1.24.4
WARNING: Ignoring invalid distribution -atplotlib (e:\python\envs\tf3\lib\site-packages)
Collecting numpy==1.24.4
  Using cached numpy-1.24.4-cp38-cp38-win_amd64.whl.metadata (5.6 kB)
Using cached numpy-1.24.4-cp38-cp38-win_amd64.whl (14.9 MB)
Installing collected packages: numpy
  Attempting uninstall: numpy
    WARNING: No metadata found in e:\python\envs\tf3\lib\site-packages
    Found existing installation: numpy 1.24.4
error: uninstall-no-record-file

× Cannot uninstall numpy 1.24.4
╰─> The package's contents are unknown: no RECORD file was found for numpy.

hint: You might be able to recover from this via: pip install --force-reinstall --no-deps numpy==1.24.4

```
所以我直接换虚拟环境了，复制了一个py38，注意要conda activate py38_copy,不然终端没变

安装完basicsrcuda消失了？不知道怎么回事了
CUDA_HOME=/usr/local/cuda \
CUDNN_INCLUDE_DIR=/usr/local/cuda \
CUDNN_LIB_DIR=/usr/local/cuda \
BASICSR_EXT=True pip install basicsr


似乎我的 
CUDA_HOME = `E:\NvidaCuda\developmet` 
CUDNN_INCLUDE_DIR = `E:\NvidaCuda\developmet\include`
CUDNN_LIB_DIR = `E:\NvidaCuda\developmet\lib\x64`

CUDA_HOME=E:\NvidaCuda\developmet
CUDNN_INCLUDE_DIR=E:\NvidaCuda\developmet\include
CUDNN_LIB_DIR=E:\NvidaCuda\developmet\lib\x64
BASICSR_EXT=True pip install basicsr


似乎是解决不了了

明白了，是basicsr偷偷把我torch改为2.4.1
我要卸载掉torch重新安装

conda install E:\GoogleDownload\pytorch-1.7.1-py3.8_cuda110_cudnn8_0.tar.bz2
同时更新torchvision conda install E:\GoogleDownload\torchvision-0.8.0-py38_cu110.tar.bz2
简简单单搞定


FileNotFoundError: [Errno 2] No such file or directory: 'experiments/001_MSRResNet_x4_f64b16_DIV2K_1000k_B16G1_wandb/models/net_g_1000000.pth'

在yml中 
path:  
pretrain_network_g: experiments/001_MSRResNet_x4_f64b16_DIV2K_1000k_B16G1_wandb/models/net_g_1000000.pth  
strict_load_g: false  
resume_state: ~ 

注释掉pretrain_network_g后的就可以


在autodl上：

注意到basicsr可以保留训练痕迹，太优雅了

autoDL上没有DF2k。。。。。那跑预训练权重得了


@ARCH_REGISTRY.register()
研究一下注册问题，先跑通代码，再看DF2k问题
我的镜像还在，明天把我的镜像分享给大号，上传数据集可以用无卡模式，感觉可以一直到需要gpu的时候

inference-推理

## SAMFN
torch=1.11

好了，终于成功了。关键在于不要通过除提供的setup.py以外的任何其他方法安装BasicSR，并确保你使用的是Python 3.8。我无意中先用Python 3.10尝试运行了这个。 
还是建立虚拟环境，然后
```
pip install -r requirements.txt
python setup.py develop
```

这种注册的模式太秒了，但是我也得想想怎么注册我的模型

![[Pasted image 20250918131022.png]]
数据是真实的

```
用这个[脚本](https://github.com/sunny2109/SAFMN/blob/main/scripts/data_preparation/extract_subimages.py)将2K图裁成480*480的patch，这样读取才快一些；  
最后，把数据集路径修改为patch所在的路径就可以了。
```
原来是要切图，其它模型应该也能这样加快训练
![[Pasted image 20250918210115.png]]
```python
    # get feature maps
    features = model(img)
    # tensor to numpy 
    features = features.squeeze().cpu().numpy()
    features = np.mean(features, axis=0)
    # show feature maps
    plt.figure()
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    plt.imshow(features)
    # plt.colorbar()
    plt.savefig('./scripts/feature_show/features/pdf/feat_input2.pdf')
    plt.close()

```




这个SAFM还是值得学习的
![[Pasted image 20250918200554.png]]


这个CCM作者没有过多说明，说是基于FMBConv改的，效果应该不错？
![[Pasted image 20250918200848.png]]

FMBConv：
![[Pasted image 20250918201236.png]]
FMBConv反bottleneck结构（也叫invert residual），类似于mobilenet，但是似乎要relu6？（有提到residual就是bottleneck在神经元少relu会丢参数）


---
明天：看camxier，准备手搓模块

## camxier
先看配置要求，创建虚拟环境
```python
conda create --name camxier python=3.8
```
终端（Shell）还不认识 `conda activate` 这个命令
```python
conda init bash
重启bash
conda activate camxier
```
下载requirement
```python
cd CAMixerSR-main
pip install -r requirements.txt
```
无卡运行内存小，编译包可能失败 
有卡也不行，在下载basicsr1.4时

![[Pasted image 20250919110621.png]]

![[Pasted image 20250919141051.png]]

代码解读：
```python
condition_wind = torch.stack(
    torch.meshgrid(torch.linspace(-1, 1, self.window_size), torch.linspace(-1, 1, self.window_size))
) \
.type_as(x).unsqueeze(0).repeat(N, 1, H // self.window_size, W // self.window_size)
```
#### 1. `torch.linspace(-1, 1, self.window_size)`

- **功能**: `linspace` 用于创建一个一维张量，其中包含在指定区间 `[start, end]` 内均匀分布的 `steps` 个点。
    
- **用法**: `torch.linspace(start, end, steps)`
    
- **在本代码中**:
    
    - `start = -1`, `end = 1`
        
    - `steps = self.window_size` (在`CAMixerSR`的配置中是16)
        
    - **作用**: 生成一个从-1到1的包含16个点的序列，作为单个窗口在一个维度（例如X轴）上的归一化坐标。
        
    - **示例输出 (假设 `window_size`=4)**: `tensor([-1.0000, -0.3333, 0.3333, 1.0000])`

#### 2. `torch.meshgrid(...)`

- **功能**: `meshgrid` 接收多个一维张量，并为它们创建一个坐标网格。
    
- **用法**: `torch.meshgrid(tensor1, tensor2, ...)`
    
- **在本代码中**:
    
    - 它接收了两个相同的 `linspace` 结果，一个代表X轴坐标，一个代表Y轴坐标。
        
    - **作用**: 生成一个 `16x16` 窗口内所有点的二维坐标网格。它会返回一个包含两个张量的元组：
        
        1. `grid_x`: X坐标网格。每一列都相同。
            
        2. `grid_y`: Y坐标网格。每一行都相同。
            
    - **示例输出 (假设 `window_size`=3, `linspace`输出为`[-1, 0, 1]`)**:
        
        - `grid_x` (X坐标):
            
            ```
            [[-1, 0, 1],
             [-1, 0, 1],
             [-1, 0, 1]]
            ```
            
        - `grid_y` (Y坐标):
            
            ```
            [[-1, -1, -1],
             [ 0,  0,  0],
             [ 1,  1,  1]]
            ```
            
        
        当你从 `grid_x` 和 `grid_y` 中取相同位置 `(i, j)` 的值时，就得到了该点在网格中的坐标，例如 `(grid_x[0,0], grid_y[0,0])` 就是 `(-1, -1)`。
        

#### 3. `torch.stack(...)`

- **功能**: `stack` 沿着一个**新的维度**将一系列张量拼接起来。这与 `torch.cat`（沿现有维度拼接）不同。
    
- **用法**: `torch.stack(tensors, dim=0)`
    
- **在本代码中**:
    
    - 它将 `meshgrid` 生成的 `grid_x` 和 `grid_y` 两个张量进行堆叠。
        
    - **作用**: 将X坐标网格和Y坐标网格合并成一个单独的张量。默认在 `dim=0` 堆叠。
        
    - **维度变化**:
        
        - 输入: 两个 `[16, 16]` 的张量。
            
        - 输出: 一个 `[2, 16, 16]` 的张量。其中，第一个通道 `[0, :, :]` 是X坐标图，第二个通道 `[1, :, :]` 是Y坐标图。这个 `[2, 16, 16]` 的张量就是**单个窗口的位置编码**。
            

#### 4. 后续处理: `.type_as(x).unsqueeze(0).repeat(...)`

现在我们有了一个代表单个窗口位置编码的张量，需要将它扩展以匹配整个特征图的尺寸 `[N, C, H, W]`。

- **`.type_as(x)`**:
    
    - **功能**: 这是一个便捷方法，它将当前张量的数据类型（如 `float32`）和设备（如 `cuda:0`）转换为与另一个张量 `x` 完全相同。
        
    - **作用**: 确保位置编码张量与模型中的其他张量在类型和设备上兼容，避免出错。
        
- **`.unsqueeze(0)`**:
    
    - **功能**: 在指定位置增加一个维度，大小为1。
        
    - **作用**: 为张量添加一个批次（batch）维度。
        
    - **维度变化**: `[2, 16, 16]` -> `[1, 2, 16, 16]` (Batch, Channels, Height, Width)
        
- **`.repeat(N, 1, H // self.window_size, W // self.window_size)`**:
    
    - **功能**: 在指定维度上重复张量内容。
        
    - **作用**: 将这个**单个**窗口的位置编码图块，像铺瓷砖一样，铺满整个特征图的空间。
        
        - `N`: 在批次维度上重复 `N` 次，以匹配输入的批次大小。
            
        - `1`: 在通道维度上不重复（保持2个通道，即X和Y）。
            
        - `H // self.window_size`: 在高度上重复 `H/16` 次。
            
        - `W // self.window_size`: 在宽度上重复 `W/16` 次。
            
    - **最终结果**: 我们得到了 `condition_wind` 张量，其维度为 `[N, 2, H, W]`。现在，特征图上任何一个点 `(i, j)` 在 `condition_wind` 中对应的值 `condition_wind[:, :, i, j]` 都存储了它在自己所属的 `16x16` 窗口内的归一化坐标。
