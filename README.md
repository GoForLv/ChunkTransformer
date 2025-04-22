# ChunkTransformer

### Todo List
1. 划分验证集
2. 数据归一化
3. 学习率调度器位置？学习率调度器在验证损失没有改善时也会更新学习率
```python
def test(self):
    # ... 现有代码 ...
    if test_loss < self.min_loss:
        self.min_loss = test_loss
        self.no_improve_epochs = 0
        self.scheduler.step(test_loss)  # 只有损失改善时才更新学习率
    else:
        self.no_improve_epochs += 1
    # ... 其余代码 ...
```
4. d_block整除问题
5. 梯度裁剪 可能导致梯度爆炸
```python
def train_epoch(self):
    # ... 现有代码 ...
    self.optimizer.step()
    # 添加梯度裁剪
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    self.optimizer.zero_grad()
    # ... 其余代码 ...
```
6. 模型保存
7. 类型提示和文档完善
8. 加入多级Pooling
9.  增加对比模型