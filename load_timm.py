import torch
import timm
import argparse

def generate_model(model, model_depth, pretrained, num_classes, n_input_channels):  
    try:#num_class must be greater than 0
        assert(num_classes > 0)
    except AssertionError as e:
        raise( AssertionError( "Additional info. %s"%e ) )
        
    #check for the model names
    possible_models = timm.list_models("*{}*".format(model))
    if pretrained:
        pretrained_models = timm.list_models(pretrained=True)
    try:#model has to be in timm
        assert(len(possible_models) > 0)
    except AssertionError as e:
        raise( AssertionError( "Additional info. %s"%e ) )
    
    #check for the model names
    final_model_name = model+str(model_depth)
    if pretrained:
        if(final_model_name in possible_models):
            if(final_model_name in pretrained_models):
                model = timm.create_model(final_model_name, pretrained=True, num_classes=num_classes)
            else:
                model = timm.create_model(final_model_name, pretrained=False, num_classes=num_classes)
        else:
            model = timm.create_model(model, pretrained=pretrained, num_classes=num_classes)
    else:
        if(final_model_name in possible_models):
            model = timm.create_model(final_model_name, pretrained=pretrained, num_classes=num_classes)
        else:
            model = timm.create_model(model, pretrained=pretrained, num_classes=num_classes)        
    
    if n_input_channels != 3:
        #get model tree
        model_tree = []
        #e.g. model.stem.conv1
        #e.g. model.conv1
        model_dict = model.__dict__["_modules"]
        key, val = list(model_dict.items())[0]
        model_tree.append(str(key))
        while type(val) != torch.nn.modules.conv.Conv2d:
            for k, v in val.named_children():
                model_tree.append(str(k))
                val = v
                break
                
        #update the first conv layer
        new_first_conv = torch.nn.Conv2d(n_input_channels, val.out_channels, val.kernel_size, val.stride, val.padding, val.dilation, val.groups, val.bias, val.padding_mode) 
        if len(model_tree) == 1:
            setattr(model, model_tree[0], new_first_conv)
        else:
            new_attr = getattr(model, model_tree[0])
            for i in range(1, len(model_tree)-1):
                new_attr2 = getattr(new_attr, model_tree[i])
                new_attr = new_attr2

            setattr(new_attr, model_tree[-1], new_first_conv)
        
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet', type=str, help='Model name')
    parser.add_argument('--model_depth', default=50, type=int, help='Depth of the model')
    parser.add_argument('--pretrained', default=True, help='If true, will use ImageNet pretrained weight')   
    parser.add_argument('--num_classes', default=2, type=int, help='number of classes')
    parser.add_argument('--n_input_channels', default=3, type=int, help='number of input channels') 
    args = parser.parse_args()
    model = generate_model(args.model, args.model_depth, args.pretrained, args.num_classes, args.n_input_channels)
    print(model)
    
if __name__ == "__main__":
    main()