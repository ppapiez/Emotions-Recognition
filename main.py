from manager import Manager
import argparse

#==================================================================================================

parser_manager = argparse.ArgumentParser(description='')
parser_manager.add_argument('--new_model', default='False')
parser_manager.add_argument('--model_architecture', default='cnn_ke_1')
parser_manager.add_argument('--output_classes', default='kaggle_classes')
parser_manager.add_argument('--path', default='./Models/cnn_ke_1_2')

args_manager = parser_manager.parse_args()


parser_training = argparse.ArgumentParser(description='')
parser_training.add_argument('--dataset', default='Kaggle-Emotions')
parser_training.add_argument('--batch_size', default=50)
parser_training.add_argument('--epochs', default=3)

args_training = parser_training.parse_args()


#==================================================================================================

if __name__ == '__main__':
    im_path = "./Data/Single images to classify/subject01_happy_small.png"

    print(args_manager)
    print(args_training)

    m = Manager(args_manager)
    m.plot_history()
    # m.train_model(args_training)
    # m.save_model()
    res, label = m.classify_image(im_path)

    print(res)
    print(label)