import argparse
import os
import sys
import time
import re

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import utils
from transformer_net import TransformerNet
from vgg import Vgg16


def check_paths(args):
    try:
        # 是否存在save_model的路径，不存在就创建一个
        # save_model_dir：将保存经过训练的模型的文件夹的路径
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        # 是否存在checkpoint的路径，不存在就创建一个
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def train(args):
    # 将torch.Tensor分配到的设备的对象CPU或GPU
    device = torch.device("cuda" if args.cuda else "cpu")
    # 初始化随机种子
    np.random.seed(args.seed)
    # 为CPU设置种子用于生成随机数
    torch.manual_seed(args.seed)
    """
        将多个transform组合起来使用
    """
    transform = transforms.Compose([
        # 重新设定大小
        transforms.Resize(args.image_size),
        # 将给定的Image进行中心切割
        transforms.CenterCrop(args.image_size),
        # 把Image转成张量Tensor格式，大小范围为[0,1]
        transforms.ToTensor(),
        # 使用lambd作为转换器
        transforms.Lambda(lambda x: x.mul(255))
    ])
    # 使用ImageFolder数据加载器，传入数据集的路径
    # transform：一个函数，原始图片作为输入，返回一个转换后的图片
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    # 把上一步做成的数据集放入Data.DataLoader中，可以生成一个迭代器
    # batch_size：int，每个batch加载多少样本
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    # 加载模型TransformerNet到设备上
    transformer = TransformerNet().to(device)
    # 我们选择Adam作为优化器
    optimizer = Adam(transformer.parameters(), args.lr)
    # 均方损失函数
    mse_loss = torch.nn.MSELoss()
    # 加载模型Vgg16到设备上
    vgg = Vgg16(requires_grad=False).to(device)
    # 风格图片的处理
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    # 载入风格图片
    style = utils.load_image(args.style_image, size=args.style_size)
    # 处理风格图片
    style = style_transform(style)
    # repeat(*sizes)沿着指定的维度重复tensor
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)
    # 特征风格归一化
    features_style = vgg(utils.normalize_batch(style))
    # 风格特征图计算Gram矩阵
    gram_style = [utils.gram_matrix(y) for y in features_style]
    # 迭代训练
    for e in range(args.epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            # 把梯度置零，也就是把loss关于weight的导数变成0
            optimizer.zero_grad()

            y = transformer(x.to(device))

            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x.cuda())
            # 计算内容损失
            content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)
            # 计算风格损失
            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= args.style_weight
            # 总损失
            total_loss = content_loss + style_loss
            # 反向传播
            total_loss.backward()
            # 更新参数
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)
            # 生成训练好的风格图片模型 and (batch_id + 1) % args.checkpoint_interval == 0
            if args.checkpoint_model_dir is not None:
                transformer.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                torch.save(transformer.state_dict(), ckpt_model_path)
                transformer.to(device).train()

    # save model
    transformer.eval().cpu()
    save_model_filename = "epoch_" + str(args.epochs) + "_" + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def stylize(args):  # 提供了一个测试，当我们训练好了模型，就可以用这个函数来帮我们生成图片了
    device = torch.device("cuda" if args.cuda else "cpu")

    content_image = utils.load_image(args.content_image, scale=args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(args.model)
        # 从checkpoint删除InstanceNorm中已保存的不建议使用的running_ *keys
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        output = style_model(content_image).cpu()
        utils.save_image(args.output_image, output[0])


def main():
    # Training settings 就是在设置一些参数，每个都有默认值
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 2")
    # batch_size参数，如果想改，如改成8可这么写：python main.py -batch_size=8
    train_arg_parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",
                                  help="path to style-image")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                  help="path to folder where checkpoints of trained models will be saved")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--style-size", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--cuda", type=int, required=True,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")
    train_arg_parser.add_argument("--style-weight", type=float, default=1e10,
                                  help="weight for style-loss, default is 1e10")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")
    # 跑多少次batch进行一次日志记录
    train_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
    train_arg_parser.add_argument("--checkpoint-interval", type=int, default=2000,
                                  help="number of batches after which a checkpoint of the trained model will be created")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image")
    eval_arg_parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")
    # 这个是使用arg_parse模块时的必备行，将参数进行关联
    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
        train(args)
    else:
        stylize(args)


if __name__ == "__main__":
    main()
