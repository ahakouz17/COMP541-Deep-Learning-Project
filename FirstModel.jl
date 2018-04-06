
using Knet

function constructFeatDict()
    # read features for all proteins
    f = open("yeast_feature_all.csv");
    lines = readlines(f);
    close(f);
    numberOfProteins = length(lines) 
    featureNames = String.(split(lines[1],",")); 
    d = length(featureNames) - 1 # number of features per protein
    proteins = lines[2:numberOfProteins];
    featuresDict = Dict{String,Any}()
    for p in proteins
        featureVect = String.(split(p, ","));
        featuresDict[featureVect[1]] = parse.(Float32, featureVect[2:d+1])
    end
    return featuresDict;
end;

function loaddata(featuresDict)
    f = open("yeast_protein_pair.csv")
    lines = readlines(f);
    close(f)
    n = length(lines); # number of samples/ protein pairs
    samples = lines[2:end - 1];
    proteinA_ID = []
    proteinA = []
    #proteinB_ID = []
    proteinB = []
    concatAB = []
    ygold = Array{UInt8,1}(n);
    i = 1;
    for s in samples
        s = String.(split(s, ","));
        # push!(proteinA_ID, s[1]);
        push!(proteinA, reshape(mat(featuresDict[s[1]]), 1, 1164));
        #push!(proteinB_ID, s[2]);
        push!(proteinB, reshape(mat(featuresDict[s[2]]), 1, 1164))
        push!(concatAB, hcat(reshape(mat(featuresDict[s[1]]), 1, 1164), reshape(mat(featuresDict[s[2]]), 1, 1164)))
        label = parse(Int64, s[3]);
        ygold[i] = convert(UInt8, label + 1);
        i += 1;
    end
    return vcat(map(Atype, concatAB)...), map(Atype, proteinA), map(Atype, proteinB), ygold
end;

# number of input features per protein
NOINPUTS = 1164;
# number of input features for the protein pair
NOCONCAT = NOINPUTS * 2;
# output is a one-hot-vector 10 -> not interacting, 01 -> intracting
NOOUTPUTS = 2;
# the number of hidden units in the hidden layers of the DeepPPI-CON model
HIDDENS = Any[NOCONCAT, 512, 256, 128, 128, NOOUTPUTS]; 

#trnper = 0.58;
trnper = 0.0001;
devper = 0.17;
tstper = 1 - trnper - devper;
NOEPOCH = 60;
BATCHSIZE = 100;
Atype = gpu() >= 0 ? KnetArray{Float32} : Array{Float32};
setseed(13);

function winit(h...)  # use winit(x,h1,h2,...,hn,y) for n hidden layer model
    w = Any[]
    for i=2:length(h)
        push!(w, xavier(h[i],h[i-1]))
        push!(w, zeros(h[i],1))
    end
    map(Atype, w)
end;

# Input X matrix and gold labels Y
# Output list of minibatches (x, y)
function minibatchi(X, Y, batchsize)
    data = Any[] # You are going to fill that data array
    for i = 1:batchsize:size(X, 2)
        bl = min(i + batchsize - 1, size(X, 2))
        push!(data, (X[:, i:bl], Y[i:bl]))
    end
    return data
end;

function dividedataset(data, ygold, trnper, devper, tstper) # 0.58, 0.17, 0.25
    nosamples = size(data,1)
    notst = Int(floor(tstper*nosamples))
    notrn = Int(floor(trnper * nosamples))
    nodev = nosamples - notrn - notst
    ind = randperm(nosamples)
  
    xtrn = data[ind[1:notrn],:];
    ytrn = ygold[ind[1:notrn]];
   
    xtst = data[ind[notrn+1:notrn+notst], :];
    ytst = ygold[ind[notrn+1:notrn+notst]];
    
    xdev = data[ind[notrn+notst+1:nosamples], :];
    ydev = ygold[ind[notrn+notst+1:nosamples]];
    
    dtrn = minibatchi(xtrn',ytrn,BATCHSIZE);
    ddev = minibatchi(xdev',ydev,BATCHSIZE);
    dtst = minibatchi(xtst',ytst,BATCHSIZE);
    
    return dtrn, ddev, dtst
end;

function predict(w,x)
    for i=1:2:length(w)
        x = w[i]*x .+ w[i+1]
        if i<length(w)-1
            x = relu.(x)   ## apply RELU to all but the final layer's output                        
        #else
          #x = x .-maximum(x, 1)
          #x = exp.(x) ./ (sum(exp.(x), 1))
        end
    end
    return x
end;

loss(w,x,ygold) = nll(predict(w,x),ygold);

lossgradient = grad(loss);

# Train model(w) with SGD and return a list containing w for every epoch
function train!(w,data,predict; epochs=100,lr=.5,o...)
    #weights = Any[deepcopy(w)]
    for epoch in 1:epochs
        for (x,y) in data
            dw = lossgradient(w,x,y)
            #print(dw[1])
            #for i in 1:length(w)
            #   w[i] -= lr * dw[i]
            #end
            update!(w,dw,lr=lr)  # w[i] = w[i] - lr * g[i]
        end
        #push!(weights,deepcopy(w))
    end
    #return weights
end;

report(epoch)=println((:epoch,epoch,:trn,accuracyi(w,dtrn,predict),:dev,accuracyi(w,ddev,predict)));

function accuracyi(ypred, ygold)
    count = 0
    for i in 1:size(ypred, 2)
        if((ypred[1,i] >= ypred[2,i] && ygold[i]==1) || (ypred[1,i] <= ypred[2,i] && ygold[i]==2))
            count +=1
        end
    end
    return count/size(ypred, 2);
end;

function accuracyi(w, data, predict)
    acc = 0;
    for (x, y) in data
        ypred = predict(w,x)
        acc += accuracyi(ypred, y) 
    end
    return acc/length(data)
end;

w = winit(HIDDENS...);
featuresDict = constructFeatDict();
concatAB, proteinA, proteinB, ygold = loaddata(featuresDict);

dtrn, ddev, dtst = dividedataset(concatAB, ygold, trnper, devper, tstper);

(x,y) = first(dtrn);
y

yp = predict(w, x)

report(0)
w = winit(HIDDENS...);
#println(w[1][2])
@time for epoch = 1:5
    train!(w, dtrn, predict; lr=0.5, epochs=1)
    display(predict(w, x))
    #report(epoch)
end

# Training Set
trnacc = 0;
for (x, y) in dtrn
    ypred = predict(w,x)
    trnacc += accuracyi(ypred, y) 
end
print(trnacc/length(dtrn))

# Test Set
tstacc = 0;
for (x, y) in dtst
    ypred = predict(w,x)
    tstacc += accuracyi(ypred, y) 
end
print(tstacc/length(dtst))

# dev Set
devacc = 0;
for (x, y) in ddev
    ypred = predict(w,x)
    devacc += accuracyi(ypred, y) 
end
print(devacc/length(ddev))
