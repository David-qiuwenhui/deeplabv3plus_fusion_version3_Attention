import os

import torch
from nets.deeplabv3_training import CE_Loss, Dice_loss, Focal_Loss, weights_init
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score


def compute_loss(
    aux_branch, inputs, focal_loss, dice_loss, target, cls_weights, num_classes, labels
):
    loss_function = Focal_Loss if focal_loss else CE_Loss
    loss = 0.0
    if aux_branch:
        # print(f"\033[1;33;44m 🔘🔘🔘🔘 使用辅助分类器计算模型的训练损失值 \033[0m")
        # ----------------------#
        #   计算主分支和辅助分类器的损失
        # ----------------------#
        # aux classification
        loss += 0.1 * loss_function(inputs.stage2_aux, target, cls_weights, num_classes)
        loss += 0.2 * loss_function(inputs.stage3_aux, target, cls_weights, num_classes)
        loss += 0.3 * loss_function(inputs.stage4_aux, target, cls_weights, num_classes)
        # main classification
        loss += 1.0 * loss_function(inputs.main, target, cls_weights, num_classes)

        if dice_loss:
            loss += 0.1 * Dice_loss(inputs.stage2_aux, labels)
            loss += 0.2 * Dice_loss(inputs.stage3_aux, labels)
            loss += 0.3 * Dice_loss(inputs.stage4_aux, labels)
            loss += 1.0 * Dice_loss(inputs.main, labels)
    else:
        # ----------------------#
        #   计算主分支的损失
        # ----------------------#
        loss += loss_function(inputs.main, target, cls_weights, num_classes)

        if dice_loss:
            loss += Dice_loss(inputs.main, labels)

    return loss


def fit_one_epoch(
    model_train,
    model,
    loss_history,
    eval_callback,
    optimizer,
    epoch,
    epoch_step,
    epoch_step_val,
    gen,
    gen_val,
    Epoch,
    cuda,
    dice_loss,
    focal_loss,
    cls_weights,
    aux_branch,
    num_classes,
    fp16,
    scaler,
    save_period,
    save_dir,
    local_rank=0,
):
    total_loss = 0
    total_f_score = 0

    val_loss = 0
    val_f_score = 0

    # ------------------------------------------------------------------#
    #   模型训练模式
    # ------------------------------------------------------------------#
    if local_rank == 0:
        print("---------- Start Train ----------")
        pbar = tqdm(
            total=epoch_step,
            desc=f"🚀Epoch {epoch + 1}/{Epoch}",
            postfix=dict,
            mininterval=0.1,
        )
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        imgs, pngs, labels = batch

        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)
        # ******************** 清零梯度 ********************
        optimizer.zero_grad()
        if not fp16:
            # ******************** 前向传播 ********************
            outputs = model_train(imgs)
            # ******************** 计算损失 ********************
            loss = compute_loss(
                aux_branch,
                outputs,
                focal_loss,
                dice_loss,
                pngs,
                weights,
                num_classes,
                labels,
            )
            assert loss is not None

            with torch.no_grad():
                # ******************** 计算f_score ********************
                _f_score = f_score(outputs.main, labels)

            # ******************** 反向传播 ********************
            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast

            # 混合精度计算
            with autocast():
                # ******************** 前向传播 ********************
                outputs = model_train(imgs)
                # ******************** 计算损失 ********************
                loss = compute_loss(
                    aux_branch,
                    outputs,
                    focal_loss,
                    dice_loss,
                    pngs,
                    weights,
                    num_classes,
                    labels,
                )
                assert loss is not None

                with torch.no_grad():
                    # ******************** 计算f_score ********************

                    _f_score = f_score(outputs.main, labels)

            # ******************** 反向传播 ********************
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        total_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(
                **{
                    "📝️train_total_loss": total_loss / (iteration + 1),
                    "📒f_score": total_f_score / (iteration + 1),
                    "📖lr": get_lr(optimizer),
                }
            )
            pbar.update(1)

    # ------------------------------------------------------------------#
    #   模型验证模式
    # ------------------------------------------------------------------#
    if local_rank == 0:
        pbar.close()
        print("--------- Finish Train ----------")
        print("********** Start Validation **********")
        pbar = tqdm(
            total=epoch_step_val,
            desc=f"💡Epoch {epoch + 1}/{Epoch}",
            postfix=dict,
            mininterval=0.1,
        )

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            # ******************** 前向传播 ********************
            outputs = model_train(imgs)

            # ******************** 计算损失 ********************
            loss = compute_loss(
                False,
                outputs,
                focal_loss,
                dice_loss,
                pngs,
                weights,
                num_classes,
                labels,
            )
            assert loss is not None

            # ******************** 计算f_score ********************
            _f_score = f_score(outputs.main, labels)

            val_loss += loss.item()
            val_f_score += _f_score.item()

            if local_rank == 0:
                pbar.set_postfix(
                    **{
                        "📝️val_loss": val_loss / (iteration + 1),
                        "📒f_score": val_f_score / (iteration + 1),
                        "📖lr": get_lr(optimizer),
                    }
                )
                pbar.update(1)

    # -------------------- 保存本次epoch的训练和验证结果 ------------------------
    if local_rank == 0:
        pbar.close()
        print("********** Finish Validation **********")
        loss_history.append_loss(
            epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val
        )
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print("Epoch:" + str(epoch + 1) + "/" + str(Epoch))
        print(
            "Total Loss: %.3f || Val Loss: %.3f "
            % (total_loss / epoch_step, val_loss / epoch_step_val)
        )

        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(
                model.state_dict(),
                os.path.join(
                    save_dir,
                    "ep%03d-loss%.3f-val_loss%.3f.pth"
                    % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val),
                ),
            )

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(
            loss_history.val_loss
        ):
            print("Save best model to best_epoch_weights.pth")
            torch.save(
                model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth")
            )

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
