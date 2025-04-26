# Training orchestration module

class ModelTrainer:
    def __init__(self, model, data_loader, criterion, optimizer):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            for inputs, targets in self.data_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}') 