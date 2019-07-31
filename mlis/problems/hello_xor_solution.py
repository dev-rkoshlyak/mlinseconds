# HelloXor is a HelloWorld of Machine Learning.
import torch
import torch.nn as nn
import torch.optim as optim

class HelloXorModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(HelloXorModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        return x

    def calc_error(self, output, target):
        # This is loss function
        return ((output-target)**2).sum()

    def calc_predict(self, output):
        # Simple round output to predict value
        return output.round()

class Solution():
    # Return trained model
    def train_model(self, train_data, train_target, context):
        # Uncommend next line to understand grid search
        print("Hint[1]: Increase hidden size")
        hidden_size = 3
        print("Hint[2]: Learning rate is too small")
        learning_rate = 1.0
        # Set up trainng parameters from context
        hidden_size, learning_rate = self.grid_search_tutorial_part1(context)
        # Model represent our neural network
        model = HelloXorModel(train_data.size(1), train_target.size(1), hidden_size)
        # Optimizer used for training neural network
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        while True:
            # Report step, so we know how many steps
            context.increase_step()
            # model.parameters()...gradient set to zero
            optimizer.zero_grad()
            # evaluate model => model.forward(data)
            output = model(train_data)
            # if x < 0.5 predict 0 else predict 1
            predict = model.calc_predict(output)
            # Number of correct predictions
            correct = predict.eq(train_target.view_as(predict)).long().sum().item()
            # Total number of needed predictions
            total = predict.view(-1).size(0)
            # No more time left or learned everything, stop training
            time_left = context.timer.get_time_left()
            if time_left < 0.1 or correct == total:
                break
            # calculate error
            error = model.calc_error(output, train_target)
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            error.backward()
            # print progress of the learning
            self.print_stats(context.step, error, correct, total)
            # update model: model.parameters() -= lr * gradient
            optimizer.step()
        # Log data for grid search
        self.grid_search_tutorial_part2(context)
        return model

    def print_stats(self, step, error, correct, total):
        if step % 1000 == 0:
            print("Step = {} Correct = {}/{} Error = {}".format(step, correct, total, error.item()))

    def grid_search_tutorial_part1(self, context):
        if context.type == context.__class__.GRID_SEARCH:
            return context.run_params.hidden_size, context.run_params.learning_rate
        else:
            print("In order to run grid search execute: python3 -m mlis.problems.hello_xor_grid_search")
            exit(0)

    def grid_search_tutorial_part2(self, context):
        # During grid search, train_model will be called runs_per_params times with every possible combination of params.
        # This can be used for automatic parameters tunning.
        if context.type == context.__class__.GRID_SEARCH:
            print("[HelloXor] hidden_size={} learning_rate={} run_idx_per_params=[{}/{}]".format(
                context.run_params.hidden_size, context.run_params.learning_rate,
                context.run_idx_per_params, context.runs_per_params))
            time_left_key = 'time_left'
            step_key = 'step'
            context.log_scalar(time_left_key, context.timer.get_time_left())
            context.log_scalar(step_key, context.step)
            if context.run_idx_per_params == context.runs_per_params-1:
                print("[HelloXor] run_params={}".format(context.run_params))
                print("[HelloXor] time_left_logs={}".format(context.get_scalars(time_left_key)))
                print("[HelloXor] step_logs={}".format(context.get_scalars(step_key)))
                print("[HelloXor] step_logs_stats={}".format(context.get_scalars_stats(step_key)))
