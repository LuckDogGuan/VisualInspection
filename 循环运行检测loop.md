项目与服务器
本地项目：D:\code\VisualInspection\铝型材缺陷图
服务器项目：/home/guanjianxiong/code/yolo/VisualInspection
SSH：ssh -F $env:USERPROFILE\.ssh\config xmu-server
服务器 Python：/opt/miniconda3/bin/python
当前自动监控：Codex heartbeat，每 5 分钟检查一次，automation id：automation
核心目标
优化铝型材 11 类分类模型。
首要目标：降低“缺陷判正常”的漏检风险。
不能只看 Top1，总准确率、弱类、漏检、误报都要一起判断。
本地只做代码和小测试，正式训练/全量评估在服务器。
模型与重要文件
Baseline 模型：/home/guanjianxiong/code/yolo/VisualInspection/outputs/classification_results/deploy/classifier.torchscript.pt
/home/guanjianxiong/code/yolo/VisualInspection/outputs/classification_results/models/model_best.pth.tar
/home/guanjianxiong/code/yolo/VisualInspection/outputs/classification_results/deploy/labels.json

计划文档：本地：[铝型材分类模型优化与未分类图片筛选计划.md](D:/code/VisualInspection/铝型材缺陷图/铝型材分类模型优化与未分类图片筛选计划.md)
服务器：/home/guanjianxiong/code/yolo/VisualInspection/铝型材分类模型优化与未分类图片筛选计划.md

供货商/数据说明：[data/ali2018/说明文件.md](D:/code/VisualInspection/铝型材缺陷图/data/ali2018/说明文件.md)

已编写代码
src/classification/run_pipeline.py支持：--class-weights
--weighted-sampler
--loss focal
--focal-gamma


src/classification/train.py已实现：FocalLoss
build_class_weights
build_weighted_sampler


src/classification/run_experiment_queue.py自动串行训练和评估实验。

src/classification/screen_raw_images.py支持 raw_images no-copy 筛选。

src/classification/compare_prediction_reports.py对比 raw_images 两次预测变化。

tests/test_training_strategies.py本地和服务器均验证过。

验证状态
本地通过：python -m py_compile src\classification\train.py src\classification\run_pipeline.py
python tests\test_training_strategies.py
python -m unittest discover -s tests
python src\classification\run_pipeline.py --dry-run --max-samples 2 --epochs 1

服务器通过：/opt/miniconda3/bin/python -m py_compile src/classification/train.py src/classification/run_pipeline.py
/opt/miniconda3/bin/python tests/test_training_strategies.py
/opt/miniconda3/bin/python src/classification/run_pipeline.py --dry-run --max-samples 2 --epochs 1

自动队列
队列脚本：src/classification/run_experiment_queue.py
队列 PID：3583964
队列日志：/home/guanjianxiong/code/yolo/VisualInspection/outputs/classification_results/jobs/experiment_queue.log

执行顺序：exp_class_weight
exp_weighted_sampler
exp_focal
exp_class_weight_focal

每个实验训练后自动生成：full_screening/summary.md
raw_images_quality/summary.md
raw_images_quality_compare.md
experiment_queue_summary.csv

Baseline 结果
Top1：92.45%
Top3：99.37%
缺陷判正常：42
正常误报：31
擦花：57.05%
碰伤：76.62%
凸粉：84.38%
脏点：87.56%
Class Weight 结果
状态：已完成，不推荐替换 Baseline。
Top1：89.59%
Top3：98.79%
缺陷判正常：39
正常误报：82
擦花：50.34%
碰伤：85.71%
凸粉：100.00%
脏点：86.12%
判断：漏检只少 3 张。
正常误报增加 51 张。
Top1 下降 2.86%。
擦花更差。
暂不推荐替换 Baseline。

当前运行状态
exp_class_weight 已完成。
exp_weighted_sampler 正在训练。
截止最后一次检查，Weighted Sampler 还没有输出 epoch。
查看命令：
cd /home/guanjianxiong/code/yolo/VisualInspection
pgrep -af 'run_experiment_queue.py|screen_labeled_dataset.py|screen_raw_images.py|compare_prediction_reports.py|run_pipeline.py --output-dir outputs/classification_results/exp'
tail -n 120 outputs/classification_results/jobs/experiment_queue.log
tail -n 120 outputs/classification_results/exp_weighted_sampler/queue_train.log
下一步
下一次 heartbeat 检查 Weighted Sampler 是否完成。
如果完成：读取 outputs/classification_results/exp_weighted_sampler/full_screening/summary.md
读取 outputs/classification_results/exp_weighted_sampler/raw_images_quality/summary.md
读取 outputs/classification_results/exp_weighted_sampler/raw_images_quality_compare.md
统计缺陷判正常和正常误报。
与 Baseline、Class Weight 对比。
更新并同步 铝型材分类模型优化与未分类图片筛选计划.md。