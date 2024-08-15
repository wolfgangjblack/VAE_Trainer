import matplotlib.pyplot as plt

def plot_save_image(tensor_value, path):
    plt.figure()
    plt.imshow(tensor_value)
    plt.savefig(path)
    plt.close()
            