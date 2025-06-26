import torch

# prediction
# Linear regression
# f = w * x  + b
# here : f = 2 * x

X = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8, 10, 12, 14, 16], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


# model output
def forward(x):
    return w * x
# its a function and x is the parameter
#You’re teaching the model: “Take an input x, multiply it by a number w, and that’s your prediction!”

# loss = MSE
def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()
# to measure prediction error.
#A smaller MSE means your predictions are close to the real values (good model).
#A larger MSE means your predictions are far off (bad model).

X_test = 5.0

print(f'Prediction before training: f({X_test}) = {forward(X_test).item():.3f}')

# Training
learning_rate = 0.01 #How fast the model learns
n_epochs = 100 #How many times we repeat the training process (epochs = training cycles).

for epoch in range(n_epochs):
    y_pred = forward(X) # the function that does the prediction
    l = loss(Y, y_pred) # the function that does the loss
    l.backward()  # # Compute gradient and store in w.grad
    #think of it as: "How much should w change to make the loss smaller?"
    # This is what autograd in PyTorch does — it finds the derivative for you.

    # You are not training anymore — you're just modifying the value of the weight w.
    with torch.no_grad():# with in python
        # “While inside this block, turn off gradient tracking. When done, turn it back on.”
        #I’m manually changing/updating weights here
        w -= learning_rate * w.grad #"Update w so that the loss gets a little smaller."
        #why w.grad?
        #Because only the update step is inside torch.no_grad().All the gradient calculation was done before that.
    w.grad.zero_() ## 💥 IMPORTANT: Clear gradient before next epoch
    if (epoch + 1) % 10 == 0:
        print(f'epoch {epoch + 1}: w = {w.item():.3f}, loss = {l.item():.3f}')

print(f'Prediction after training: f({X_test}) = {forward(X_test).item():.3f}')

