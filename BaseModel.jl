
using Knet

NOINPUTS = 1164;
NOCONCAT = NOINPUTS * 2;
NOOUTPUTS = 2;
HIDDENS = Any[512, 256, 128, 128, 2]; 
NOEPOCH = 60;
BATCHSIZE = 100;
Atype =  Array{Float32};
#gpu() >= 0 ? KnetArray{Float32} :

w = map(Atype, [ randn(512, NOCONCAT),  zeros(512, 1), 
                  randn(256, 512), zeros(256, 1),
                  randn(128,256),  zeros(128, 1),
                  randn(128,128),  zeros(128, 1),
                 randn(2,128), zeros(2,1)]);

function predict(w,x)
    for i=1:2:length(w)
        x = w[i]*x .+ w[i+1]
        if i<length(w)-1
            x = max.(0,x)   ## apply RELU to all but the final layer's output                        
        end
    end
    return x
end

# read features for all proteins
f = open("yeast_feature_all.csv")
lines = readlines(f);
close(f)

numberOfProteins = length(lines) #number of proteins
featureNames = String.(split(lines[1],",")); # names of the features
d = length(featureNames) - 1 # number of features per protein
proteins = lines[2:numberOfProteins];
featuresDict = Dict{String,Any}()
for p in proteins
    featureVect = String.(split(p, ","));
    featuresDict[featureVect[1]] = parse.(Float32, featureVect[2:d+1])
end

f = open("yeast_protein_pair.csv")
lines = readlines(f);
close(f)

n = length(lines); # number of samples/ protein pairs
featureNames = String.(split(lines[1],",")); # names of the features
samples = lines[2:n];

proteinA_ID = []
proteinA = []
proteinB_ID = []
proteinB = []
concatAB = []
ygold = []
for s in samples
    s = String.(split(s, ","));
    push!(proteinA_ID, s[1]);
    push!(proteinA, reshape(mat(featuresDict[s[1]]), 1, 1164));
    push!(proteinB_ID, s[2]);
    push!(proteinB, reshape(mat(featuresDict[s[2]]), 1, 1164))
    push!(concatAB, hcat(reshape(mat(featuresDict[s[1]]), 1, 1164), reshape(mat(featuresDict[s[2]]), 1, 1164)))
    push!(ygold, parse(Int64, s[3]));
end

# Input X matrix and gold labels Y
# Output list of minibatches (x, y)
function minibatchi(X, Y, batchsize)
    data = Any[] # You are going to fill that data array
    # YOUR CODE STARTS HERE
    for i = 1:batchsize:size(X, 2)
        bl = min(i + batchsize - 1, size(X, 2))
        push!(data, (X[:, i:bl], Y[:, i:bl]))
    end
    #YOUR CODE ENDS HERE
    return data
end

nosamples = size(mat(concatAB),1)
notst = Int(floor(0.25*nosamples))
notrn = Int(floor(0.75 * (nosamples-notst)))
nodev = nosamples - notrn - notst
ind = randperm(nosamples)

xtrni = concatAB[ind[1:notrn]];
xtrn = vcat(xtrni...)';
ytrn = mat(ygold[ind[1:notrn]])';

xtsti = concatAB[ind[notrn+1:notrn+notst]];
xtst = vcat(xtsti...)';
ytst = mat(ygold[ind[notrn+1:notrn+notst]])';

xdevi = concatAB[ind[notrn+notst+1:nosamples]];
xdev = vcat(xdevi...)';
ydev = mat(ygold[ind[notrn+notst+1:nosamples]])';

dtrn = minibatch(xtrn,ytrn,BATCHSIZE);
ddev = minibatch(xdev,ydev,BATCHSIZE);
dtst = minibatch(xtst,ytst,BATCHSIZE);

function accuracy(ypred, ygold)
    count = 0
    for i in 1:size(ypred, 2)
        if(ypred[1,i] >= ypred[2,i] && ygold[i]==0)
            count +=1
        end
    end
    return count/size(ypred, 2);
end

# Training Set
ypred = predict(w,xtrn)
accuracy(ypred, ytrn)

# Test Set
ytstpred = predict(w,xtst)
accuracy(ytstpred, ytst)

# Validation Set
ydevpred = predict(w,xdev)
accuracy(ydevpred, ydev)
