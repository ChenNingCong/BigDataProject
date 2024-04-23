from multitask import *
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
writer = SummaryWriter("runs/multitask")
task_niter = {task_id:0 for task_id in all_task_ids}
total_niter = 0
for epoch in range(num_epochs):
    model.train()
    for batch in tqdm(data_loader):
        batch = to_device_collator(batch)
        output = model(**batch)
        optimizers[task_id].zero_grad()
        model_optmizer.zero_grad()
        loss = output.loss
        loss.backward()
        task_id = batch["task_id"]
        task_niter[task_id] += 1
        optimizers[task_id].step()
        model_optmizer.step()
        lr_schedulers[task_id].step()
        model_lr_scheduler.step()  
        
        total_niter += 1
        writer.add_scalar(f"{task_id}_loss", loss.item(), task_niter[task_id])
        writer.add_scalar(f"{task_id}_lr", lr_schedulers[task_id].get_last_lr()[0], task_niter[task_id])
        writer.add_scalar(f"model_lr", model_lr_scheduler.get_last_lr()[0], task_niter[task_id])
        writer.add_scalar("total_loss", loss.item(), total_niter)

    result = evaluate()
    for task_id in result:
        for metric in result[task_id]:
            writer.add_scalar(f"{task_id}_{metric}", result[task_id][metric], epoch)
