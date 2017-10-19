class MultiClothNet():
    def __init__(self):
        self.pixels_per_cm = 4
        self.center = [305, 705] # Adjust these values
        self.net = ClothPresentNet()

    def predict(image):
        predictions = []
        for i in range(-30, 30, 20):
            for j in range(-60, 60, 20):
                target_x = i*self.pixels_per_cm + self.center[0]
                target_y = j*self.pixels_per_cm + self.center[1]
                prediction = self.net.predict(image[target_x-30:target_x+30, target_y-30:target_y+30])
                if prediction[1] > prediction[0]:
                    predictions.append((i, j))
        return predictions
