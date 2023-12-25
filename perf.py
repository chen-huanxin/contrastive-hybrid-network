import torch
from torchstat import stat

from tools.model_maker import test_model

iterations = 300   # 重复计算的轮次
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

def perf_test(args, model):
    model.eval()
    data = torch.randn(args.batch_size, args.img_dim, args.reshape, args.reshape).to(args.device)
    # GPU预热
    for _ in range(50):
        # _ = model(data)
        _  = test_model(args, model, data)

    # 测速
    times = torch.zeros(iterations)     # 存储每轮iteration的时间

    with torch.no_grad():
        for iter in range(iterations):
            starter.record()
            # _ = model(data)
            _ = test_model(args, model, data)
            ender.record()
            # 同步GPU时间
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) # 计算时间
            times[iter] = curr_time
            # print(curr_time)n

    mean_time = times.mean().item()
    args.logger.info("Inference time: {:.6f}ms, FPS: {} ".format(mean_time, 1000/mean_time))


def stat_test(args, model):
    model = model.cpu()
    stat(model, (args.img_dim, args.reshape, args.reshape))


if __name__ == "__main__":
    pass