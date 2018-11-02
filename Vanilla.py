import torch
import random

hm_vectors = 4
vector_size = 13


max_prop_time = 20


default_layers = (20,10,15)

default_filters = (
    (1, 2, 3, 4),
    (5, 6, 7, 8, 9),
    (9, 10, 11)
                   )

default_loss_multipliers = (1, 0.05, 0.05, 0.05)


#   Structure


def create_model(filters=default_filters,layers=default_layers):

    model = []
    hm_filters = len(filters)
    layers = layers + tuple([vector_size*(hm_vectors-1)])

    # layer : presence_near

    model.append(
        {
            'vr':torch.randn([1,hm_vectors], requires_grad=True),
            'ur':torch.randn([vector_size,vector_size], requires_grad=True),
            'br':torch.zeros([1,vector_size], requires_grad=True),

            'vf':torch.randn([1,hm_vectors], requires_grad=True),
            'uf':torch.randn([vector_size,vector_size], requires_grad=True),
            'bf':torch.zeros([1,vector_size], requires_grad=True),

            'va':torch.randn([1,hm_vectors], requires_grad=True),
            'ua':torch.randn([vector_size,vector_size], requires_grad=True),
            'ba':torch.zeros([1,vector_size], requires_grad=True),

            'vs':torch.randn([1,hm_vectors], requires_grad=True),
            'us':torch.randn([vector_size,vector_size], requires_grad=True),
            'bs':torch.zeros([1,vector_size], requires_grad=True),
        }
    )

    # layer : convolution

    layer = {}
    for _,filter in enumerate(filters):
        layer['wf'+str(_)] = torch.randn([len(filter),1], requires_grad=True)

    model.append(layer)

    # layer : presence_far

    in_size = vector_size*hm_filters

    model.append(
        {
            'vr':torch.randn([1,hm_filters], requires_grad=True),
            'vr2':torch.randn([in_size,vector_size], requires_grad=True),
            'ur':torch.randn([vector_size,vector_size], requires_grad=True),
            'br':torch.zeros([1,vector_size], requires_grad=True),

            'va':torch.randn([1,hm_filters], requires_grad=True),
            'va2':torch.randn([in_size,vector_size], requires_grad=True),
            'ua':torch.randn([vector_size,vector_size], requires_grad=True),
            'ba':torch.zeros([1,vector_size], requires_grad=True),

            'vs':torch.randn([1,hm_filters], requires_grad=True),
            'vs2':torch.randn([in_size,vector_size], requires_grad=True),
            'bs':torch.zeros([1,vector_size], requires_grad=True),
        }
    )

    # layer : decision

    for _,layer_size in enumerate(layers):

        if _ == 0:
            in_size = vector_size*(hm_vectors-1)
            out_size = layer_size
        else:
            in_size = layers[_-1]
            out_size = layer_size

        layer = {
            'vr':torch.randn([in_size,out_size], requires_grad=True),
            'ur':torch.randn([layer_size,layer_size], requires_grad=True),
            'br':torch.zeros([1,out_size], requires_grad=True),

            'vf':torch.randn([in_size,out_size], requires_grad=True),
            'uf':torch.randn([layer_size,layer_size], requires_grad=True),
            'bf':torch.zeros([1,out_size], requires_grad=True),

            'va':torch.randn([in_size,out_size], requires_grad=True),
            'ua':torch.randn([layer_size,layer_size], requires_grad=True),
            'ba':torch.zeros([1,out_size], requires_grad=True),

            'vs':torch.randn([in_size,out_size], requires_grad=True),
            'us':torch.randn([layer_size,layer_size], requires_grad=True),
            'bs':torch.zeros([1,out_size], requires_grad=True),
        }

        if _ == len(layers)-1:

            layer['vr2'] = torch.randn([vector_size,layer_size], requires_grad=True)
            layer['vf2'] = torch.randn([vector_size,layer_size], requires_grad=True)
            layer['va2'] = torch.randn([vector_size,layer_size], requires_grad=True)
            layer['vs2'] = torch.randn([vector_size,layer_size], requires_grad=True)


        model.append(layer)

    # for _ in range(1,hm_vectors):
    #
    #     str_ = '_'+str(_)
    #
    #     layer['vr'+str_] = torch.randn([vector_size,1], requires_grad=True)
    #     layer['ur'+str_] = torch.randn([vector_size, vector_size], requires_grad=True)
    #
    #     layer['va'+str_] = torch.randn([vector_size,1], requires_grad=True)
    #     layer['ua'+str_] = torch.randn([vector_size, vector_size], requires_grad=True)
    #
    #     layer['vf'+str_] = torch.randn([vector_size,1], requires_grad=True)
    #     layer['uf'+str_] = torch.randn([vector_size, vector_size], requires_grad=True)
    #
    #     layer['vs'+str_] = torch.randn([vector_size, 1], requires_grad=True)
    #     layer['us'+str_] = torch.randn([vector_size,vector_size], requires_grad=True)
    #


    return model



# Forward Prop


def forward_prop(model, sequence, context=None, gen_seed=None, gen_iterations=None, filters=default_filters, dropout=0.0):


    #   listen


    states = [context] if context is not None else init_states(model)
    outputs = []

    for t in range(len(sequence[0])):

        output, state = prop_timestep(model, sequence[t], states[-1], filters=filters, dropout=dropout)

        outputs.append(output)
        states.append(state)


    #   generate


    states = [states[-1]]
    outputs = [gen_seed] if gen_seed is not None else [outputs[-1]]     # sequence, an array of outputs, where out[0] = vocab, out[1] = rhythm

    if gen_iterations is None:

        t = 0
        while t < max_prop_time and not stop_cond(outputs[-1]):

            output, state = prop_timestep(model, outputs[-1], states[-1], filters=filters, dropout=0.0)

            outputs.append(output)
            states.append(state)
            t += 1

    else:

        for t in range(gen_iterations):

            output, state = prop_timestep(model, outputs[-1], states[-1], filters=filters, dropout=dropout)

            outputs.append(output)
            states.append(state)


    del outputs[0]
    return outputs


def prop_timestep(model, sequence_t, context_t, filters, dropout):

    produced_outputs = []
    produced_context = []

    # layer : presence_near

    input = torch.stack(sequence_t,0)
    # input = torch.cat(sequence_t)
    # input = torch.zeros([1, len(input)], requires_grad=False) + input

    remember = torch.sigmoid(
        torch.matmul(model[0]['vr'],input) +
        torch.matmul(context_t[0],model[0]['ur']) +
        model[0]['br']
    )

    forget = torch.sigmoid(
        torch.matmul(model[0]['vf'],input) +
        torch.matmul(context_t[0],model[0]['uf']) +
        model[0]['bf']
    )

    attention = torch.tanh(
        torch.matmul(model[0]['va'],input) +
        torch.matmul(context_t[0],model[0]['ua']) +
        model[0]['ba']
    )

    short_mem = torch.tanh(
        torch.matmul(model[0]['vs'],input) +
        torch.matmul(context_t[0],model[0]['us']) +
        model[0]['bs']
    )

    state = remember * short_mem + forget * context_t[0]
    state0_focused = (attention * torch.tanh(state))

    # layer_produced_context.append()
    # produced_outputs.append((attention * torch.tanh(layer_produced_context[-1])).squeeze(dim=0))

    if dropout != 0.0:
        drop = random.choices(range(len(state)), k=int(len(state) * dropout))
        for _ in drop: state[_] = 0

    produced_context.append(state)

    # layer : convolution

    input = state0_focused

    convolutions = [vector_convolve(input, filter_values=filter, filter_weights=model[1]['wf'+str(_)])
                    for _, filter in enumerate(filters)]

    # layer : presence_far

    input = torch.stack(convolutions,0)
    input2 = torch.cat(convolutions)
    input2 = torch.zeros([1, len(input2)], requires_grad=False) + input2

    remember = torch.sigmoid(
        torch.matmul(model[2]['vr'], input) +
        torch.matmul(input2, model[2]['vr2']) +
        torch.matmul(context_t[1], model[2]['ur']) +
        model[2]['br']
    )

    attention = torch.tanh(
        torch.matmul(model[2]['va'], input) +
        torch.matmul(input2, model[2]['va2']) +
        torch.matmul(context_t[1], model[2]['ua']) +
        model[2]['ba']
    )

    short_mem = torch.tanh(
        torch.matmul(model[2]['vs'], input) +
        torch.matmul(input2, model[2]['vs2']) +
        attention * context_t[1] +
        model[2]['bs']
    )

    state = remember * short_mem + (1-remember) * context_t[1]

    produced_outputs.append(state.squeeze(dim=0))

    if dropout != 0.0:
        drop = random.choices(range(len(state)), k=int(len(state) * dropout))
        for _ in drop: state[_] = 0

    produced_context.append(state)

    # layer : decision

    decision_outputs = []

    hm_layers = len(model)-3

    for _ in range(hm_layers-1):

        layer = model[_+3]
        state = context_t[_+2]
        input = decision_outputs[-1] if _ != 0 else torch.cat(sequence_t[1:])

        remember = torch.sigmoid(
            torch.matmul(input, layer['vr']) +
            torch.matmul(state, layer['ur'])
        )

        forget = torch.sigmoid(
            torch.matmul(input, layer['vf']) +
            torch.matmul(state, layer['uf'])
        )

        attention = torch.tanh(
            torch.matmul(input, layer['vr']) +
            torch.matmul(state, layer['ur'])
        )

        short_mem = torch.tanh(
            torch.matmul(input, layer['vs']) +
            torch.matmul(state, layer['us'])
        )

        state = remember * short_mem + forget * state
        produced_context.append(state)
        output = (attention * torch.tanh(produced_context[-1])).squeeze(dim=0)
        decision_outputs.append(output)

        # decision _finalize

    input = decision_outputs[-1]
    input2 = torch.tensor(produced_outputs[0], requires_grad=False)
    state = context_t[-1]
    layer = model[-1]

    remember = torch.sigmoid(
        torch.matmul(input, layer['vr']) +
        torch.matmul(input2, layer['vr2']) +
        torch.matmul(state, layer['ur']) +
        layer['br']
    )

    forget = torch.sigmoid(
        torch.matmul(input, layer['vf']) +
        torch.matmul(input2, layer['vf2']) +
        torch.matmul(state, layer['uf']) +
        layer['bf']
    )

    attention = torch.tanh(
        torch.matmul(input, layer['va']) +
        torch.matmul(input2, layer['va2']) +
        torch.matmul(state, layer['ua']) +
        layer['ba']
    )

    short_mem = torch.tanh(
        torch.matmul(input, layer['vs']) +
        torch.matmul(input2, layer['vs2']) +
        torch.matmul(state, layer['us']) +
        layer['bs']
    )

    state = remember * short_mem + forget * state
    produced_context.append(state)
    output = (attention * torch.tanh(produced_context[-1])).squeeze(dim=0)
    produced_outputs.append(output[:vector_size])
    produced_outputs.append(output[vector_size:vector_size*2])
    produced_outputs.append(output[-vector_size:])


    return produced_outputs, produced_context


#   Math Operations


def vector_convolve(vector, filter_values, filter_weights):

    to_convolve = []
    convolution = []

    for _ in range(vector_size):
        conv_vect = [vector[:,(_+i) % vector_size] for i in filter_values]
        to_convolve.append(torch.cat(conv_vect))

    for element in to_convolve:
        convolution.append((element * filter_weights).sum())

    return torch.Tensor(convolution)


def loss_wrt_distance(output_seq, label_seq):

    sequence_losses = [[],[],[],[]]

    for t in range(len(label_seq)):
        lbl = label_seq[t]
        pred = output_seq[t]

        for _,lbl_e in enumerate(lbl):
            pred_e = pred[_]

            loss = (lbl_e - pred_e).pow(2)
            # loss = lbl_e - pred_e
            # loss = lbl_e * -(torch.log(pred_e)) if lbl_e != 0 else 0

            sequence_losses[_].append(loss.sum())
            # if _ == 0 : sequence_losses[_].append(loss.sum())

    return sequence_losses


#   Optimization


def update_gradients(sequence_loss, loss_multipliers=default_loss_multipliers):
    for _,node in enumerate(sequence_loss):
        multiplier = [loss_multipliers[_]]
        for time_step in node:
           time_step.backward(retain_graph=True,
                              gradient=torch.Tensor(multiplier)
                              )


def update_model(model, batch_size=1, learning_rate=0.001):
    with torch.no_grad():
        for _,layer in enumerate(model):
            for __,weight in enumerate(layer.values()):
                if weight.grad is not None:
                    weight -= learning_rate * weight.grad / batch_size
                    weight.grad = None


def update_model_momentum(model, moments, batch_size=1, alpha=0.9, beta=0.1):
    with torch.no_grad():
        for _,layer in enumerate(model):
            for __,weight in enumerate(layer.values()):
                if weight.grad is not None:
                    weight.grad /= batch_size
                    moments[_][__] = alpha * moments[_][__] + beta * weight.grad
                    weight -= moments[_][__]
                    weight.grad = None


def update_model_rmsprop(model, accu_grads, batch_size=1, lr=0.01, alpha=0.9):
    with torch.no_grad():
        for _,layer in enumerate(model):
            for __,weight in enumerate(layer.values()):
                if weight.grad is not None:
                    weight.grad /= batch_size
                    accu_grads[_][__] = alpha * accu_grads[_][__] + (1 - alpha) * weight.grad**2
                    weight -= lr * weight.grad / (torch.sqrt(accu_grads[_][__]) + 1e-8)
                    weight.grad = None


def update_model_adam(model, accugrads, moments, epoch_nr, batch_size=1, lr=0.001, alpha_moments=0.9, alpha_accugrads=0.999):
    with torch.no_grad():
        for _,layer in enumerate(model):
            for __,weight in enumerate(layer.values()):
                if weight.grad is not None:
                    weight.grad /= batch_size

                    moments[_][__] = alpha_moments * moments[_][__] + (1 - alpha_moments) * weight.grad
                    accugrads[_][__] = alpha_accugrads * accugrads[_][__] + (1 - alpha_accugrads) * weight.grad ** 2

                    moment_hat = moments[_][__] / (1 - alpha_moments ** epoch_nr)
                    accugrad_hat = accugrads[_][__] / (1 - alpha_accugrads ** epoch_nr)

                    weight -= lr * moment_hat / (torch.sqrt(sum(accugrad_hat)) + 1e-8)
                    weight.grad = None


#   Helpers


def init_states(model):

    states_t0 = [torch.randn([1,vector_size], requires_grad=True) for _ in range(2)]    # presence _near & _far

    [states_t0.append(torch.randn([1,layer['vr'].size()[1]], requires_grad=True))       # decision deep-layer(s)
        for layer in model[3:]]

    return [states_t0]


stop_dur = 2.0


def stop_cond(output_t):

    durations = output_t[2]

    for dur in durations:
        if float(dur) >= stop_dur:
            return True
    return False


def return_grads(model):
    grads = []
    for _,layer in enumerate(model):
        for w in layer:
            grads.append(layer[w].grad)
    return grads


def apply_grads(model, grads):
    ctr = 0
    for _,layer in enumerate(model):
        for w in layer:
            this_grad = grads[ctr]
            if layer[w].grad is None: layer[w].grad = this_grad
            else: layer[w].grad += this_grad

            ctr +=1
