from classes import load_checkpoint, predict
from input_args import input_args
from data import process_image
import json

in_arg = input_args()

with open(in_arg.class_values, 'r') as f:
    class_values = json.load(f)

def main():
      
    model = load_checkpoint(in_arg.filepath)
    
    ps, classes = predict(in_arg.img_path, model, in_arg.top_k, in_arg.gpu)
    
    
    print('Top K Probabilities:', list(ps))
    print('Top K Classes:',list(classes))

    top_options = [class_values[x] for x in classes]
    
    print('Most likely class: ', top_options[0].capitalize(), 'with a probability of:', round(ps[0] * 100, 2),'%' )
    
    
    
if __name__ == "__main__":
    main()    