# t5-pegasus-chinese
基于GOOGLE T5中文生成式模型的摘要生成/指代消解，支持batch批量生成，多进程

**如果你想了解自己是否需要本Git，请看如下几点介绍（重点）：**
1. 模型可部署在CPU/GPU，均测试可用
2. 基于谷歌t5的中文生成式预训练模型
3. 集成了中文摘要生成、指代消解等生成任务语料，开箱即用
4. 基于PyTorch
5. 支持多张显卡DataParallel
6. 支持batch批量推理/生成
7. 支持多进程，推理/生成提速

**本 Git 如何运行：**  
1. 所需Python库  
    - transformers==4.15.0  
    - tokeniziers==0.10.3  
    - torch==1.7.0或1.8.0或1.8.1均可
    - jieba
    - rouge
    - tqdm
    - pandas 
2. 下载t5-pegasus模型放在 t5_pegasus_pretain目录下，目录下三个文件：
   - pytorch_model.bin
   - config.json
   - vocab.txt  

    预训练模型下载地址（追一科技开源的t5-pegasus的pytorch版本，分享自renmada）：
    - Base版本：
      - 百度网盘：https://pan.baidu.com/s/1JIjEEyX-dgmqpQdL7aNbAw 提取码：fd7k
      - Google Drive：https://drive.google.com/file/d/18Y5LVghAGbz7ys0noii1eM1yDtFmW490/view?usp=sharing
    - Small版本：
      - 百度网盘：https://pan.baidu.com/s/1Kc6xFqJZoVxKLBGx924zgQ 提取码：hvq9
      - Google Drive：https://drive.google.com/file/d/1fCL7f_f8I6YuezoYQ3EvT9h-S1WngKZo/view?usp=sharing

    解压后，按上面说的放在对应目录下，文件名称确认无误即可。
3. 命令行执行
   - 训练finetune
        ```bash
        python train_with_finetune.py
        ```
   - 预测generate
        ```bash
        python predict_with_generate.py
        ```
   - 预测generate(多进程，仅支持Linux系统，Windows系统不可用)
        ```bash
        python predict_with_generate.py --use_multiprocess
        ```
**语料介绍：**

**t5-pegasus模型的细节，以便了解它为什么能在摘要任务中有效:**

**实验结果：**


**如对本Git内容存有疑问或建议，欢迎在issue区或者邮箱isguanjing@126.com与我联系。**
