Җ(
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02v2.7.0-0-gc256c071bb28??%
z
dense_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_26/kernel
s
#dense_26/kernel/Read/ReadVariableOpReadVariableOpdense_26/kernel*
_output_shapes

:*
dtype0
r
dense_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_26/bias
k
!dense_26/bias/Read/ReadVariableOpReadVariableOpdense_26/bias*
_output_shapes
:*
dtype0
z
dense_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_27/kernel
s
#dense_27/kernel/Read/ReadVariableOpReadVariableOpdense_27/kernel*
_output_shapes

:*
dtype0
r
dense_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_27/bias
k
!dense_27/bias/Read/ReadVariableOpReadVariableOpdense_27/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
7token_and_position_embedding_14/embedding_32/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*H
shared_name97token_and_position_embedding_14/embedding_32/embeddings
?
Ktoken_and_position_embedding_14/embedding_32/embeddings/Read/ReadVariableOpReadVariableOp7token_and_position_embedding_14/embedding_32/embeddings*
_output_shapes
:	?*
dtype0
?
7token_and_position_embedding_14/embedding_33/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?N*H
shared_name97token_and_position_embedding_14/embedding_33/embeddings
?
Ktoken_and_position_embedding_14/embedding_33/embeddings/Read/ReadVariableOpReadVariableOp7token_and_position_embedding_14/embedding_33/embeddings*
_output_shapes
:	?N*
dtype0
?
:transformer_block_3/multi_head_attention_3/dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*K
shared_name<:transformer_block_3/multi_head_attention_3/dense_20/kernel
?
Ntransformer_block_3/multi_head_attention_3/dense_20/kernel/Read/ReadVariableOpReadVariableOp:transformer_block_3/multi_head_attention_3/dense_20/kernel*
_output_shapes

:*
dtype0
?
8transformer_block_3/multi_head_attention_3/dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8transformer_block_3/multi_head_attention_3/dense_20/bias
?
Ltransformer_block_3/multi_head_attention_3/dense_20/bias/Read/ReadVariableOpReadVariableOp8transformer_block_3/multi_head_attention_3/dense_20/bias*
_output_shapes
:*
dtype0
?
:transformer_block_3/multi_head_attention_3/dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*K
shared_name<:transformer_block_3/multi_head_attention_3/dense_21/kernel
?
Ntransformer_block_3/multi_head_attention_3/dense_21/kernel/Read/ReadVariableOpReadVariableOp:transformer_block_3/multi_head_attention_3/dense_21/kernel*
_output_shapes

:*
dtype0
?
8transformer_block_3/multi_head_attention_3/dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8transformer_block_3/multi_head_attention_3/dense_21/bias
?
Ltransformer_block_3/multi_head_attention_3/dense_21/bias/Read/ReadVariableOpReadVariableOp8transformer_block_3/multi_head_attention_3/dense_21/bias*
_output_shapes
:*
dtype0
?
:transformer_block_3/multi_head_attention_3/dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*K
shared_name<:transformer_block_3/multi_head_attention_3/dense_22/kernel
?
Ntransformer_block_3/multi_head_attention_3/dense_22/kernel/Read/ReadVariableOpReadVariableOp:transformer_block_3/multi_head_attention_3/dense_22/kernel*
_output_shapes

:*
dtype0
?
8transformer_block_3/multi_head_attention_3/dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8transformer_block_3/multi_head_attention_3/dense_22/bias
?
Ltransformer_block_3/multi_head_attention_3/dense_22/bias/Read/ReadVariableOpReadVariableOp8transformer_block_3/multi_head_attention_3/dense_22/bias*
_output_shapes
:*
dtype0
?
:transformer_block_3/multi_head_attention_3/dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*K
shared_name<:transformer_block_3/multi_head_attention_3/dense_23/kernel
?
Ntransformer_block_3/multi_head_attention_3/dense_23/kernel/Read/ReadVariableOpReadVariableOp:transformer_block_3/multi_head_attention_3/dense_23/kernel*
_output_shapes

:*
dtype0
?
8transformer_block_3/multi_head_attention_3/dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8transformer_block_3/multi_head_attention_3/dense_23/bias
?
Ltransformer_block_3/multi_head_attention_3/dense_23/bias/Read/ReadVariableOpReadVariableOp8transformer_block_3/multi_head_attention_3/dense_23/bias*
_output_shapes
:*
dtype0
z
dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_24/kernel
s
#dense_24/kernel/Read/ReadVariableOpReadVariableOpdense_24/kernel*
_output_shapes

:*
dtype0
r
dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_24/bias
k
!dense_24/bias/Read/ReadVariableOpReadVariableOpdense_24/bias*
_output_shapes
:*
dtype0
z
dense_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_25/kernel
s
#dense_25/kernel/Read/ReadVariableOpReadVariableOpdense_25/kernel*
_output_shapes

:*
dtype0
r
dense_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_25/bias
k
!dense_25/bias/Read/ReadVariableOpReadVariableOpdense_25/bias*
_output_shapes
:*
dtype0
?
/transformer_block_3/layer_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/transformer_block_3/layer_normalization_6/gamma
?
Ctransformer_block_3/layer_normalization_6/gamma/Read/ReadVariableOpReadVariableOp/transformer_block_3/layer_normalization_6/gamma*
_output_shapes
:*
dtype0
?
.transformer_block_3/layer_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.transformer_block_3/layer_normalization_6/beta
?
Btransformer_block_3/layer_normalization_6/beta/Read/ReadVariableOpReadVariableOp.transformer_block_3/layer_normalization_6/beta*
_output_shapes
:*
dtype0
?
/transformer_block_3/layer_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/transformer_block_3/layer_normalization_7/gamma
?
Ctransformer_block_3/layer_normalization_7/gamma/Read/ReadVariableOpReadVariableOp/transformer_block_3/layer_normalization_7/gamma*
_output_shapes
:*
dtype0
?
.transformer_block_3/layer_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.transformer_block_3/layer_normalization_7/beta
?
Btransformer_block_3/layer_normalization_7/beta/Read/ReadVariableOpReadVariableOp.transformer_block_3/layer_normalization_7/beta*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/dense_26/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_26/kernel/m
?
*Adam/dense_26/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_26/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_26/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_26/bias/m
y
(Adam/dense_26/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_26/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_27/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_27/kernel/m
?
*Adam/dense_27/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_27/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_27/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_27/bias/m
y
(Adam/dense_27/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_27/bias/m*
_output_shapes
:*
dtype0
?
>Adam/token_and_position_embedding_14/embedding_32/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*O
shared_name@>Adam/token_and_position_embedding_14/embedding_32/embeddings/m
?
RAdam/token_and_position_embedding_14/embedding_32/embeddings/m/Read/ReadVariableOpReadVariableOp>Adam/token_and_position_embedding_14/embedding_32/embeddings/m*
_output_shapes
:	?*
dtype0
?
>Adam/token_and_position_embedding_14/embedding_33/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?N*O
shared_name@>Adam/token_and_position_embedding_14/embedding_33/embeddings/m
?
RAdam/token_and_position_embedding_14/embedding_33/embeddings/m/Read/ReadVariableOpReadVariableOp>Adam/token_and_position_embedding_14/embedding_33/embeddings/m*
_output_shapes
:	?N*
dtype0
?
AAdam/transformer_block_3/multi_head_attention_3/dense_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*R
shared_nameCAAdam/transformer_block_3/multi_head_attention_3/dense_20/kernel/m
?
UAdam/transformer_block_3/multi_head_attention_3/dense_20/kernel/m/Read/ReadVariableOpReadVariableOpAAdam/transformer_block_3/multi_head_attention_3/dense_20/kernel/m*
_output_shapes

:*
dtype0
?
?Adam/transformer_block_3/multi_head_attention_3/dense_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*P
shared_nameA?Adam/transformer_block_3/multi_head_attention_3/dense_20/bias/m
?
SAdam/transformer_block_3/multi_head_attention_3/dense_20/bias/m/Read/ReadVariableOpReadVariableOp?Adam/transformer_block_3/multi_head_attention_3/dense_20/bias/m*
_output_shapes
:*
dtype0
?
AAdam/transformer_block_3/multi_head_attention_3/dense_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*R
shared_nameCAAdam/transformer_block_3/multi_head_attention_3/dense_21/kernel/m
?
UAdam/transformer_block_3/multi_head_attention_3/dense_21/kernel/m/Read/ReadVariableOpReadVariableOpAAdam/transformer_block_3/multi_head_attention_3/dense_21/kernel/m*
_output_shapes

:*
dtype0
?
?Adam/transformer_block_3/multi_head_attention_3/dense_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*P
shared_nameA?Adam/transformer_block_3/multi_head_attention_3/dense_21/bias/m
?
SAdam/transformer_block_3/multi_head_attention_3/dense_21/bias/m/Read/ReadVariableOpReadVariableOp?Adam/transformer_block_3/multi_head_attention_3/dense_21/bias/m*
_output_shapes
:*
dtype0
?
AAdam/transformer_block_3/multi_head_attention_3/dense_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*R
shared_nameCAAdam/transformer_block_3/multi_head_attention_3/dense_22/kernel/m
?
UAdam/transformer_block_3/multi_head_attention_3/dense_22/kernel/m/Read/ReadVariableOpReadVariableOpAAdam/transformer_block_3/multi_head_attention_3/dense_22/kernel/m*
_output_shapes

:*
dtype0
?
?Adam/transformer_block_3/multi_head_attention_3/dense_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*P
shared_nameA?Adam/transformer_block_3/multi_head_attention_3/dense_22/bias/m
?
SAdam/transformer_block_3/multi_head_attention_3/dense_22/bias/m/Read/ReadVariableOpReadVariableOp?Adam/transformer_block_3/multi_head_attention_3/dense_22/bias/m*
_output_shapes
:*
dtype0
?
AAdam/transformer_block_3/multi_head_attention_3/dense_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*R
shared_nameCAAdam/transformer_block_3/multi_head_attention_3/dense_23/kernel/m
?
UAdam/transformer_block_3/multi_head_attention_3/dense_23/kernel/m/Read/ReadVariableOpReadVariableOpAAdam/transformer_block_3/multi_head_attention_3/dense_23/kernel/m*
_output_shapes

:*
dtype0
?
?Adam/transformer_block_3/multi_head_attention_3/dense_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*P
shared_nameA?Adam/transformer_block_3/multi_head_attention_3/dense_23/bias/m
?
SAdam/transformer_block_3/multi_head_attention_3/dense_23/bias/m/Read/ReadVariableOpReadVariableOp?Adam/transformer_block_3/multi_head_attention_3/dense_23/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_24/kernel/m
?
*Adam/dense_24/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_24/bias/m
y
(Adam/dense_24/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_25/kernel/m
?
*Adam/dense_25/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_25/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_25/bias/m
y
(Adam/dense_25/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_25/bias/m*
_output_shapes
:*
dtype0
?
6Adam/transformer_block_3/layer_normalization_6/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86Adam/transformer_block_3/layer_normalization_6/gamma/m
?
JAdam/transformer_block_3/layer_normalization_6/gamma/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_3/layer_normalization_6/gamma/m*
_output_shapes
:*
dtype0
?
5Adam/transformer_block_3/layer_normalization_6/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75Adam/transformer_block_3/layer_normalization_6/beta/m
?
IAdam/transformer_block_3/layer_normalization_6/beta/m/Read/ReadVariableOpReadVariableOp5Adam/transformer_block_3/layer_normalization_6/beta/m*
_output_shapes
:*
dtype0
?
6Adam/transformer_block_3/layer_normalization_7/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86Adam/transformer_block_3/layer_normalization_7/gamma/m
?
JAdam/transformer_block_3/layer_normalization_7/gamma/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_3/layer_normalization_7/gamma/m*
_output_shapes
:*
dtype0
?
5Adam/transformer_block_3/layer_normalization_7/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75Adam/transformer_block_3/layer_normalization_7/beta/m
?
IAdam/transformer_block_3/layer_normalization_7/beta/m/Read/ReadVariableOpReadVariableOp5Adam/transformer_block_3/layer_normalization_7/beta/m*
_output_shapes
:*
dtype0
?
Adam/dense_26/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_26/kernel/v
?
*Adam/dense_26/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_26/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_26/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_26/bias/v
y
(Adam/dense_26/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_26/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_27/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_27/kernel/v
?
*Adam/dense_27/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_27/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_27/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_27/bias/v
y
(Adam/dense_27/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_27/bias/v*
_output_shapes
:*
dtype0
?
>Adam/token_and_position_embedding_14/embedding_32/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*O
shared_name@>Adam/token_and_position_embedding_14/embedding_32/embeddings/v
?
RAdam/token_and_position_embedding_14/embedding_32/embeddings/v/Read/ReadVariableOpReadVariableOp>Adam/token_and_position_embedding_14/embedding_32/embeddings/v*
_output_shapes
:	?*
dtype0
?
>Adam/token_and_position_embedding_14/embedding_33/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?N*O
shared_name@>Adam/token_and_position_embedding_14/embedding_33/embeddings/v
?
RAdam/token_and_position_embedding_14/embedding_33/embeddings/v/Read/ReadVariableOpReadVariableOp>Adam/token_and_position_embedding_14/embedding_33/embeddings/v*
_output_shapes
:	?N*
dtype0
?
AAdam/transformer_block_3/multi_head_attention_3/dense_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*R
shared_nameCAAdam/transformer_block_3/multi_head_attention_3/dense_20/kernel/v
?
UAdam/transformer_block_3/multi_head_attention_3/dense_20/kernel/v/Read/ReadVariableOpReadVariableOpAAdam/transformer_block_3/multi_head_attention_3/dense_20/kernel/v*
_output_shapes

:*
dtype0
?
?Adam/transformer_block_3/multi_head_attention_3/dense_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*P
shared_nameA?Adam/transformer_block_3/multi_head_attention_3/dense_20/bias/v
?
SAdam/transformer_block_3/multi_head_attention_3/dense_20/bias/v/Read/ReadVariableOpReadVariableOp?Adam/transformer_block_3/multi_head_attention_3/dense_20/bias/v*
_output_shapes
:*
dtype0
?
AAdam/transformer_block_3/multi_head_attention_3/dense_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*R
shared_nameCAAdam/transformer_block_3/multi_head_attention_3/dense_21/kernel/v
?
UAdam/transformer_block_3/multi_head_attention_3/dense_21/kernel/v/Read/ReadVariableOpReadVariableOpAAdam/transformer_block_3/multi_head_attention_3/dense_21/kernel/v*
_output_shapes

:*
dtype0
?
?Adam/transformer_block_3/multi_head_attention_3/dense_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*P
shared_nameA?Adam/transformer_block_3/multi_head_attention_3/dense_21/bias/v
?
SAdam/transformer_block_3/multi_head_attention_3/dense_21/bias/v/Read/ReadVariableOpReadVariableOp?Adam/transformer_block_3/multi_head_attention_3/dense_21/bias/v*
_output_shapes
:*
dtype0
?
AAdam/transformer_block_3/multi_head_attention_3/dense_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*R
shared_nameCAAdam/transformer_block_3/multi_head_attention_3/dense_22/kernel/v
?
UAdam/transformer_block_3/multi_head_attention_3/dense_22/kernel/v/Read/ReadVariableOpReadVariableOpAAdam/transformer_block_3/multi_head_attention_3/dense_22/kernel/v*
_output_shapes

:*
dtype0
?
?Adam/transformer_block_3/multi_head_attention_3/dense_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*P
shared_nameA?Adam/transformer_block_3/multi_head_attention_3/dense_22/bias/v
?
SAdam/transformer_block_3/multi_head_attention_3/dense_22/bias/v/Read/ReadVariableOpReadVariableOp?Adam/transformer_block_3/multi_head_attention_3/dense_22/bias/v*
_output_shapes
:*
dtype0
?
AAdam/transformer_block_3/multi_head_attention_3/dense_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*R
shared_nameCAAdam/transformer_block_3/multi_head_attention_3/dense_23/kernel/v
?
UAdam/transformer_block_3/multi_head_attention_3/dense_23/kernel/v/Read/ReadVariableOpReadVariableOpAAdam/transformer_block_3/multi_head_attention_3/dense_23/kernel/v*
_output_shapes

:*
dtype0
?
?Adam/transformer_block_3/multi_head_attention_3/dense_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*P
shared_nameA?Adam/transformer_block_3/multi_head_attention_3/dense_23/bias/v
?
SAdam/transformer_block_3/multi_head_attention_3/dense_23/bias/v/Read/ReadVariableOpReadVariableOp?Adam/transformer_block_3/multi_head_attention_3/dense_23/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_24/kernel/v
?
*Adam/dense_24/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_24/bias/v
y
(Adam/dense_24/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_25/kernel/v
?
*Adam/dense_25/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_25/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_25/bias/v
y
(Adam/dense_25/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_25/bias/v*
_output_shapes
:*
dtype0
?
6Adam/transformer_block_3/layer_normalization_6/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86Adam/transformer_block_3/layer_normalization_6/gamma/v
?
JAdam/transformer_block_3/layer_normalization_6/gamma/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_3/layer_normalization_6/gamma/v*
_output_shapes
:*
dtype0
?
5Adam/transformer_block_3/layer_normalization_6/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75Adam/transformer_block_3/layer_normalization_6/beta/v
?
IAdam/transformer_block_3/layer_normalization_6/beta/v/Read/ReadVariableOpReadVariableOp5Adam/transformer_block_3/layer_normalization_6/beta/v*
_output_shapes
:*
dtype0
?
6Adam/transformer_block_3/layer_normalization_7/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86Adam/transformer_block_3/layer_normalization_7/gamma/v
?
JAdam/transformer_block_3/layer_normalization_7/gamma/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_3/layer_normalization_7/gamma/v*
_output_shapes
:*
dtype0
?
5Adam/transformer_block_3/layer_normalization_7/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75Adam/transformer_block_3/layer_normalization_7/beta/v
?
IAdam/transformer_block_3/layer_normalization_7/beta/v/Read/ReadVariableOpReadVariableOp5Adam/transformer_block_3/layer_normalization_7/beta/v*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ۋ
valueЋB̋ Bċ
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
n
	token_emb
pos_emb
	variables
trainable_variables
regularization_losses
	keras_api
?
att
ffn

layernorm1

layernorm2
dropout1
dropout2
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
 trainable_variables
!regularization_losses
"	keras_api
R
#	variables
$trainable_variables
%regularization_losses
&	keras_api
h

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
R
-	variables
.trainable_variables
/regularization_losses
0	keras_api
h

1kernel
2bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
?
7iter

8beta_1

9beta_2
	:decay
;learning_rate'm?(m?1m?2m?<m?=m?>m??m?@m?Am?Bm?Cm?Dm?Em?Fm?Gm?Hm?Im?Jm?Km?Lm?Mm?'v?(v?1v?2v?<v?=v?>v??v?@v?Av?Bv?Cv?Dv?Ev?Fv?Gv?Hv?Iv?Jv?Kv?Lv?Mv?
?
<0
=1
>2
?3
@4
A5
B6
C7
D8
E9
F10
G11
H12
I13
J14
K15
L16
M17
'18
(19
120
221
?
<0
=1
>2
?3
@4
A5
B6
C7
D8
E9
F10
G11
H12
I13
J14
K15
L16
M17
'18
(19
120
221
 
?
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics

	variables
trainable_variables
regularization_losses
 
b
<
embeddings
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
b
=
embeddings
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api

<0
=1

<0
=1
 
?
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses
?
`query_dense
a	key_dense
bvalue_dense
	cdense
d	variables
etrainable_variables
fregularization_losses
g	keras_api
?
hlayer_with_weights-0
hlayer-0
ilayer_with_weights-1
ilayer-1
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
q
naxis
	Jgamma
Kbeta
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
q
saxis
	Lgamma
Mbeta
t	variables
utrainable_variables
vregularization_losses
w	keras_api
R
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
R
|	variables
}trainable_variables
~regularization_losses
	keras_api
v
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15
v
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
 trainable_variables
!regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
#	variables
$trainable_variables
%regularization_losses
[Y
VARIABLE_VALUEdense_26/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_26/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1

'0
(1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
)	variables
*trainable_variables
+regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
-	variables
.trainable_variables
/regularization_losses
[Y
VARIABLE_VALUEdense_27/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_27/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

10
21

10
21
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
3	variables
4trainable_variables
5regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE7token_and_position_embedding_14/embedding_32/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE7token_and_position_embedding_14/embedding_33/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE:transformer_block_3/multi_head_attention_3/dense_20/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE8transformer_block_3/multi_head_attention_3/dense_20/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE:transformer_block_3/multi_head_attention_3/dense_21/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE8transformer_block_3/multi_head_attention_3/dense_21/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE:transformer_block_3/multi_head_attention_3/dense_22/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE8transformer_block_3/multi_head_attention_3/dense_22/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE:transformer_block_3/multi_head_attention_3/dense_23/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE8transformer_block_3/multi_head_attention_3/dense_23/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_24/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_24/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_25/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_25/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/transformer_block_3/layer_normalization_6/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.transformer_block_3/layer_normalization_6/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/transformer_block_3/layer_normalization_7/gamma'variables/16/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.transformer_block_3/layer_normalization_7/beta'variables/17/.ATTRIBUTES/VARIABLE_VALUE
 
8
0
1
2
3
4
5
6
7

?0
 
 

<0

<0
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses

=0

=0
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
 

0
1
 
 
 
l

>kernel
?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

@kernel
Abias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

Bkernel
Cbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

Dkernel
Ebias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
8
>0
?1
@2
A3
B4
C5
D6
E7
8
>0
?1
@2
A3
B4
C5
D6
E7
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
d	variables
etrainable_variables
fregularization_losses
l

Fkernel
Gbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

Hkernel
Ibias
?	variables
?trainable_variables
?regularization_losses
?	keras_api

F0
G1
H2
I3

F0
G1
H2
I3
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
j	variables
ktrainable_variables
lregularization_losses
 

J0
K1

J0
K1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
o	variables
ptrainable_variables
qregularization_losses
 

L0
M1

L0
M1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
t	variables
utrainable_variables
vregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
x	variables
ytrainable_variables
zregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
|	variables
}trainable_variables
~regularization_losses
 
*
0
1
2
3
4
5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 

>0
?1

>0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

@0
A1

@0
A1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

B0
C1

B0
C1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

D0
E1

D0
E1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 

`0
a1
b2
c3
 
 
 

F0
G1

F0
G1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

H0
I1

H0
I1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 

h0
i1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
~|
VARIABLE_VALUEAdam/dense_26/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_26/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_27/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_27/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE>Adam/token_and_position_embedding_14/embedding_32/embeddings/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE>Adam/token_and_position_embedding_14/embedding_33/embeddings/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAAdam/transformer_block_3/multi_head_attention_3/dense_20/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE?Adam/transformer_block_3/multi_head_attention_3/dense_20/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAAdam/transformer_block_3/multi_head_attention_3/dense_21/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE?Adam/transformer_block_3/multi_head_attention_3/dense_21/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAAdam/transformer_block_3/multi_head_attention_3/dense_22/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE?Adam/transformer_block_3/multi_head_attention_3/dense_22/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAAdam/transformer_block_3/multi_head_attention_3/dense_23/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE?Adam/transformer_block_3/multi_head_attention_3/dense_23/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_24/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_24/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_25/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_25/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/transformer_block_3/layer_normalization_6/gamma/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE5Adam/transformer_block_3/layer_normalization_6/beta/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/transformer_block_3/layer_normalization_7/gamma/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE5Adam/transformer_block_3/layer_normalization_7/beta/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_26/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_26/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_27/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_27/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE>Adam/token_and_position_embedding_14/embedding_32/embeddings/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE>Adam/token_and_position_embedding_14/embedding_33/embeddings/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAAdam/transformer_block_3/multi_head_attention_3/dense_20/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE?Adam/transformer_block_3/multi_head_attention_3/dense_20/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAAdam/transformer_block_3/multi_head_attention_3/dense_21/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE?Adam/transformer_block_3/multi_head_attention_3/dense_21/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAAdam/transformer_block_3/multi_head_attention_3/dense_22/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE?Adam/transformer_block_3/multi_head_attention_3/dense_22/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAAdam/transformer_block_3/multi_head_attention_3/dense_23/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE?Adam/transformer_block_3/multi_head_attention_3/dense_23/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_24/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_24/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_25/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_25/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/transformer_block_3/layer_normalization_6/gamma/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE5Adam/transformer_block_3/layer_normalization_6/beta/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/transformer_block_3/layer_normalization_7/gamma/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE5Adam/transformer_block_3/layer_normalization_7/beta/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_17Placeholder*,
_output_shapes
:??????????'*
dtype0*!
shape:??????????'
?	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_177token_and_position_embedding_14/embedding_33/embeddings7token_and_position_embedding_14/embedding_32/embeddings:transformer_block_3/multi_head_attention_3/dense_20/kernel8transformer_block_3/multi_head_attention_3/dense_20/bias:transformer_block_3/multi_head_attention_3/dense_21/kernel8transformer_block_3/multi_head_attention_3/dense_21/bias:transformer_block_3/multi_head_attention_3/dense_22/kernel8transformer_block_3/multi_head_attention_3/dense_22/bias:transformer_block_3/multi_head_attention_3/dense_23/kernel8transformer_block_3/multi_head_attention_3/dense_23/bias/transformer_block_3/layer_normalization_6/gamma.transformer_block_3/layer_normalization_6/betadense_24/kerneldense_24/biasdense_25/kerneldense_25/bias/transformer_block_3/layer_normalization_7/gamma.transformer_block_3/layer_normalization_7/betadense_26/kerneldense_26/biasdense_27/kerneldense_27/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_18181
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?&
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_26/kernel/Read/ReadVariableOp!dense_26/bias/Read/ReadVariableOp#dense_27/kernel/Read/ReadVariableOp!dense_27/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpKtoken_and_position_embedding_14/embedding_32/embeddings/Read/ReadVariableOpKtoken_and_position_embedding_14/embedding_33/embeddings/Read/ReadVariableOpNtransformer_block_3/multi_head_attention_3/dense_20/kernel/Read/ReadVariableOpLtransformer_block_3/multi_head_attention_3/dense_20/bias/Read/ReadVariableOpNtransformer_block_3/multi_head_attention_3/dense_21/kernel/Read/ReadVariableOpLtransformer_block_3/multi_head_attention_3/dense_21/bias/Read/ReadVariableOpNtransformer_block_3/multi_head_attention_3/dense_22/kernel/Read/ReadVariableOpLtransformer_block_3/multi_head_attention_3/dense_22/bias/Read/ReadVariableOpNtransformer_block_3/multi_head_attention_3/dense_23/kernel/Read/ReadVariableOpLtransformer_block_3/multi_head_attention_3/dense_23/bias/Read/ReadVariableOp#dense_24/kernel/Read/ReadVariableOp!dense_24/bias/Read/ReadVariableOp#dense_25/kernel/Read/ReadVariableOp!dense_25/bias/Read/ReadVariableOpCtransformer_block_3/layer_normalization_6/gamma/Read/ReadVariableOpBtransformer_block_3/layer_normalization_6/beta/Read/ReadVariableOpCtransformer_block_3/layer_normalization_7/gamma/Read/ReadVariableOpBtransformer_block_3/layer_normalization_7/beta/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_26/kernel/m/Read/ReadVariableOp(Adam/dense_26/bias/m/Read/ReadVariableOp*Adam/dense_27/kernel/m/Read/ReadVariableOp(Adam/dense_27/bias/m/Read/ReadVariableOpRAdam/token_and_position_embedding_14/embedding_32/embeddings/m/Read/ReadVariableOpRAdam/token_and_position_embedding_14/embedding_33/embeddings/m/Read/ReadVariableOpUAdam/transformer_block_3/multi_head_attention_3/dense_20/kernel/m/Read/ReadVariableOpSAdam/transformer_block_3/multi_head_attention_3/dense_20/bias/m/Read/ReadVariableOpUAdam/transformer_block_3/multi_head_attention_3/dense_21/kernel/m/Read/ReadVariableOpSAdam/transformer_block_3/multi_head_attention_3/dense_21/bias/m/Read/ReadVariableOpUAdam/transformer_block_3/multi_head_attention_3/dense_22/kernel/m/Read/ReadVariableOpSAdam/transformer_block_3/multi_head_attention_3/dense_22/bias/m/Read/ReadVariableOpUAdam/transformer_block_3/multi_head_attention_3/dense_23/kernel/m/Read/ReadVariableOpSAdam/transformer_block_3/multi_head_attention_3/dense_23/bias/m/Read/ReadVariableOp*Adam/dense_24/kernel/m/Read/ReadVariableOp(Adam/dense_24/bias/m/Read/ReadVariableOp*Adam/dense_25/kernel/m/Read/ReadVariableOp(Adam/dense_25/bias/m/Read/ReadVariableOpJAdam/transformer_block_3/layer_normalization_6/gamma/m/Read/ReadVariableOpIAdam/transformer_block_3/layer_normalization_6/beta/m/Read/ReadVariableOpJAdam/transformer_block_3/layer_normalization_7/gamma/m/Read/ReadVariableOpIAdam/transformer_block_3/layer_normalization_7/beta/m/Read/ReadVariableOp*Adam/dense_26/kernel/v/Read/ReadVariableOp(Adam/dense_26/bias/v/Read/ReadVariableOp*Adam/dense_27/kernel/v/Read/ReadVariableOp(Adam/dense_27/bias/v/Read/ReadVariableOpRAdam/token_and_position_embedding_14/embedding_32/embeddings/v/Read/ReadVariableOpRAdam/token_and_position_embedding_14/embedding_33/embeddings/v/Read/ReadVariableOpUAdam/transformer_block_3/multi_head_attention_3/dense_20/kernel/v/Read/ReadVariableOpSAdam/transformer_block_3/multi_head_attention_3/dense_20/bias/v/Read/ReadVariableOpUAdam/transformer_block_3/multi_head_attention_3/dense_21/kernel/v/Read/ReadVariableOpSAdam/transformer_block_3/multi_head_attention_3/dense_21/bias/v/Read/ReadVariableOpUAdam/transformer_block_3/multi_head_attention_3/dense_22/kernel/v/Read/ReadVariableOpSAdam/transformer_block_3/multi_head_attention_3/dense_22/bias/v/Read/ReadVariableOpUAdam/transformer_block_3/multi_head_attention_3/dense_23/kernel/v/Read/ReadVariableOpSAdam/transformer_block_3/multi_head_attention_3/dense_23/bias/v/Read/ReadVariableOp*Adam/dense_24/kernel/v/Read/ReadVariableOp(Adam/dense_24/bias/v/Read/ReadVariableOp*Adam/dense_25/kernel/v/Read/ReadVariableOp(Adam/dense_25/bias/v/Read/ReadVariableOpJAdam/transformer_block_3/layer_normalization_6/gamma/v/Read/ReadVariableOpIAdam/transformer_block_3/layer_normalization_6/beta/v/Read/ReadVariableOpJAdam/transformer_block_3/layer_normalization_7/gamma/v/Read/ReadVariableOpIAdam/transformer_block_3/layer_normalization_7/beta/v/Read/ReadVariableOpConst*V
TinO
M2K	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_20069
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_26/kerneldense_26/biasdense_27/kerneldense_27/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate7token_and_position_embedding_14/embedding_32/embeddings7token_and_position_embedding_14/embedding_33/embeddings:transformer_block_3/multi_head_attention_3/dense_20/kernel8transformer_block_3/multi_head_attention_3/dense_20/bias:transformer_block_3/multi_head_attention_3/dense_21/kernel8transformer_block_3/multi_head_attention_3/dense_21/bias:transformer_block_3/multi_head_attention_3/dense_22/kernel8transformer_block_3/multi_head_attention_3/dense_22/bias:transformer_block_3/multi_head_attention_3/dense_23/kernel8transformer_block_3/multi_head_attention_3/dense_23/biasdense_24/kerneldense_24/biasdense_25/kerneldense_25/bias/transformer_block_3/layer_normalization_6/gamma.transformer_block_3/layer_normalization_6/beta/transformer_block_3/layer_normalization_7/gamma.transformer_block_3/layer_normalization_7/betatotalcountAdam/dense_26/kernel/mAdam/dense_26/bias/mAdam/dense_27/kernel/mAdam/dense_27/bias/m>Adam/token_and_position_embedding_14/embedding_32/embeddings/m>Adam/token_and_position_embedding_14/embedding_33/embeddings/mAAdam/transformer_block_3/multi_head_attention_3/dense_20/kernel/m?Adam/transformer_block_3/multi_head_attention_3/dense_20/bias/mAAdam/transformer_block_3/multi_head_attention_3/dense_21/kernel/m?Adam/transformer_block_3/multi_head_attention_3/dense_21/bias/mAAdam/transformer_block_3/multi_head_attention_3/dense_22/kernel/m?Adam/transformer_block_3/multi_head_attention_3/dense_22/bias/mAAdam/transformer_block_3/multi_head_attention_3/dense_23/kernel/m?Adam/transformer_block_3/multi_head_attention_3/dense_23/bias/mAdam/dense_24/kernel/mAdam/dense_24/bias/mAdam/dense_25/kernel/mAdam/dense_25/bias/m6Adam/transformer_block_3/layer_normalization_6/gamma/m5Adam/transformer_block_3/layer_normalization_6/beta/m6Adam/transformer_block_3/layer_normalization_7/gamma/m5Adam/transformer_block_3/layer_normalization_7/beta/mAdam/dense_26/kernel/vAdam/dense_26/bias/vAdam/dense_27/kernel/vAdam/dense_27/bias/v>Adam/token_and_position_embedding_14/embedding_32/embeddings/v>Adam/token_and_position_embedding_14/embedding_33/embeddings/vAAdam/transformer_block_3/multi_head_attention_3/dense_20/kernel/v?Adam/transformer_block_3/multi_head_attention_3/dense_20/bias/vAAdam/transformer_block_3/multi_head_attention_3/dense_21/kernel/v?Adam/transformer_block_3/multi_head_attention_3/dense_21/bias/vAAdam/transformer_block_3/multi_head_attention_3/dense_22/kernel/v?Adam/transformer_block_3/multi_head_attention_3/dense_22/bias/vAAdam/transformer_block_3/multi_head_attention_3/dense_23/kernel/v?Adam/transformer_block_3/multi_head_attention_3/dense_23/bias/vAdam/dense_24/kernel/vAdam/dense_24/bias/vAdam/dense_25/kernel/vAdam/dense_25/bias/v6Adam/transformer_block_3/layer_normalization_6/gamma/v5Adam/transformer_block_3/layer_normalization_6/beta/v6Adam/transformer_block_3/layer_normalization_7/gamma/v5Adam/transformer_block_3/layer_normalization_7/beta/v*U
TinN
L2J*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_20298??"
?
V
:__inference_global_average_pooling1d_3_layer_call_fn_19503

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_global_average_pooling1d_3_layer_call_and_return_conditional_losses_17301`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????':T P
,
_output_shapes
:??????????'
 
_user_specified_nameinputs
?
?
'__inference_model_1_layer_call_fn_18014
input_17
unknown:	?N
	unknown_0:	?
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_17unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_17918o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:??????????': : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:??????????'
"
_user_specified_name
input_17
?-
?	
B__inference_model_1_layer_call_and_return_conditional_losses_17918

inputs8
%token_and_position_embedding_14_17866:	?N8
%token_and_position_embedding_14_17868:	?+
transformer_block_3_17871:'
transformer_block_3_17873:+
transformer_block_3_17875:'
transformer_block_3_17877:+
transformer_block_3_17879:'
transformer_block_3_17881:+
transformer_block_3_17883:'
transformer_block_3_17885:'
transformer_block_3_17887:'
transformer_block_3_17889:+
transformer_block_3_17891:'
transformer_block_3_17893:+
transformer_block_3_17895:'
transformer_block_3_17897:'
transformer_block_3_17899:'
transformer_block_3_17901: 
dense_26_17906:
dense_26_17908: 
dense_27_17912:
dense_27_17914:
identity?? dense_26/StatefulPartitionedCall? dense_27/StatefulPartitionedCall?"dropout_10/StatefulPartitionedCall?"dropout_11/StatefulPartitionedCall?7token_and_position_embedding_14/StatefulPartitionedCall?+transformer_block_3/StatefulPartitionedCall?
7token_and_position_embedding_14/StatefulPartitionedCallStatefulPartitionedCallinputs%token_and_position_embedding_14_17866%token_and_position_embedding_14_17868*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *c
f^R\
Z__inference_token_and_position_embedding_14_layer_call_and_return_conditional_losses_17012?
+transformer_block_3/StatefulPartitionedCallStatefulPartitionedCall@token_and_position_embedding_14/StatefulPartitionedCall:output:0transformer_block_3_17871transformer_block_3_17873transformer_block_3_17875transformer_block_3_17877transformer_block_3_17879transformer_block_3_17881transformer_block_3_17883transformer_block_3_17885transformer_block_3_17887transformer_block_3_17889transformer_block_3_17891transformer_block_3_17893transformer_block_3_17895transformer_block_3_17897transformer_block_3_17899transformer_block_3_17901*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????'*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_transformer_block_3_layer_call_and_return_conditional_losses_17767?
*global_average_pooling1d_3/PartitionedCallPartitionedCall4transformer_block_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_global_average_pooling1d_3_layer_call_and_return_conditional_losses_17301?
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_10_layer_call_and_return_conditional_losses_17461?
 dense_26/StatefulPartitionedCallStatefulPartitionedCall+dropout_10/StatefulPartitionedCall:output:0dense_26_17906dense_26_17908*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_26_layer_call_and_return_conditional_losses_17321?
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_17428?
 dense_27/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0dense_27_17912dense_27_17914*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_27_layer_call_and_return_conditional_losses_17344x
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall8^token_and_position_embedding_14/StatefulPartitionedCall,^transformer_block_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:??????????': : : : : : : : : : : : : : : : : : : : : : 2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2r
7token_and_position_embedding_14/StatefulPartitionedCall7token_and_position_embedding_14/StatefulPartitionedCall2Z
+transformer_block_3/StatefulPartitionedCall+transformer_block_3/StatefulPartitionedCall:T P
,
_output_shapes
:??????????'
 
_user_specified_nameinputs
?	
d
E__inference_dropout_10_layer_call_and_return_conditional_losses_19542

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_10_layer_call_and_return_conditional_losses_17308

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
q
U__inference_global_average_pooling1d_3_layer_call_and_return_conditional_losses_17301

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :g
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????U
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????':T P
,
_output_shapes
:??????????'
 
_user_specified_nameinputs
?
?
?__inference_token_and_position_embedding_14_layer_call_fn_18888
x
unknown:	?N
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *c
f^R\
Z__inference_token_and_position_embedding_14_layer_call_and_return_conditional_losses_17012t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????'`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????': : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
,
_output_shapes
:??????????'

_user_specified_namex
?
?
(__inference_dense_24_layer_call_fn_19757

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_16808t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????'`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????': : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????'
 
_user_specified_nameinputs
?-
?	
B__inference_model_1_layer_call_and_return_conditional_losses_18124
input_178
%token_and_position_embedding_14_18072:	?N8
%token_and_position_embedding_14_18074:	?+
transformer_block_3_18077:'
transformer_block_3_18079:+
transformer_block_3_18081:'
transformer_block_3_18083:+
transformer_block_3_18085:'
transformer_block_3_18087:+
transformer_block_3_18089:'
transformer_block_3_18091:'
transformer_block_3_18093:'
transformer_block_3_18095:+
transformer_block_3_18097:'
transformer_block_3_18099:+
transformer_block_3_18101:'
transformer_block_3_18103:'
transformer_block_3_18105:'
transformer_block_3_18107: 
dense_26_18112:
dense_26_18114: 
dense_27_18118:
dense_27_18120:
identity?? dense_26/StatefulPartitionedCall? dense_27/StatefulPartitionedCall?"dropout_10/StatefulPartitionedCall?"dropout_11/StatefulPartitionedCall?7token_and_position_embedding_14/StatefulPartitionedCall?+transformer_block_3/StatefulPartitionedCall?
7token_and_position_embedding_14/StatefulPartitionedCallStatefulPartitionedCallinput_17%token_and_position_embedding_14_18072%token_and_position_embedding_14_18074*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *c
f^R\
Z__inference_token_and_position_embedding_14_layer_call_and_return_conditional_losses_17012?
+transformer_block_3/StatefulPartitionedCallStatefulPartitionedCall@token_and_position_embedding_14/StatefulPartitionedCall:output:0transformer_block_3_18077transformer_block_3_18079transformer_block_3_18081transformer_block_3_18083transformer_block_3_18085transformer_block_3_18087transformer_block_3_18089transformer_block_3_18091transformer_block_3_18093transformer_block_3_18095transformer_block_3_18097transformer_block_3_18099transformer_block_3_18101transformer_block_3_18103transformer_block_3_18105transformer_block_3_18107*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????'*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_transformer_block_3_layer_call_and_return_conditional_losses_17767?
*global_average_pooling1d_3/PartitionedCallPartitionedCall4transformer_block_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_global_average_pooling1d_3_layer_call_and_return_conditional_losses_17301?
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_10_layer_call_and_return_conditional_losses_17461?
 dense_26/StatefulPartitionedCallStatefulPartitionedCall+dropout_10/StatefulPartitionedCall:output:0dense_26_18112dense_26_18114*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_26_layer_call_and_return_conditional_losses_17321?
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_17428?
 dense_27/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0dense_27_18118dense_27_18120*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_27_layer_call_and_return_conditional_losses_17344x
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall8^token_and_position_embedding_14/StatefulPartitionedCall,^transformer_block_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:??????????': : : : : : : : : : : : : : : : : : : : : : 2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2r
7token_and_position_embedding_14/StatefulPartitionedCall7token_and_position_embedding_14/StatefulPartitionedCall2Z
+transformer_block_3/StatefulPartitionedCall+transformer_block_3/StatefulPartitionedCall:V R
,
_output_shapes
:??????????'
"
_user_specified_name
input_17
?
?
(__inference_dense_27_layer_call_fn_19598

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_27_layer_call_and_return_conditional_losses_17344o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
N__inference_transformer_block_3_layer_call_and_return_conditional_losses_17262

inputsS
Amulti_head_attention_3_dense_20_tensordot_readvariableop_resource:M
?multi_head_attention_3_dense_20_biasadd_readvariableop_resource:S
Amulti_head_attention_3_dense_21_tensordot_readvariableop_resource:M
?multi_head_attention_3_dense_21_biasadd_readvariableop_resource:S
Amulti_head_attention_3_dense_22_tensordot_readvariableop_resource:M
?multi_head_attention_3_dense_22_biasadd_readvariableop_resource:S
Amulti_head_attention_3_dense_23_tensordot_readvariableop_resource:M
?multi_head_attention_3_dense_23_biasadd_readvariableop_resource:I
;layer_normalization_6_batchnorm_mul_readvariableop_resource:E
7layer_normalization_6_batchnorm_readvariableop_resource:I
7sequential_3_dense_24_tensordot_readvariableop_resource:C
5sequential_3_dense_24_biasadd_readvariableop_resource:I
7sequential_3_dense_25_tensordot_readvariableop_resource:C
5sequential_3_dense_25_biasadd_readvariableop_resource:I
;layer_normalization_7_batchnorm_mul_readvariableop_resource:E
7layer_normalization_7_batchnorm_readvariableop_resource:
identity??.layer_normalization_6/batchnorm/ReadVariableOp?2layer_normalization_6/batchnorm/mul/ReadVariableOp?.layer_normalization_7/batchnorm/ReadVariableOp?2layer_normalization_7/batchnorm/mul/ReadVariableOp?6multi_head_attention_3/dense_20/BiasAdd/ReadVariableOp?8multi_head_attention_3/dense_20/Tensordot/ReadVariableOp?6multi_head_attention_3/dense_21/BiasAdd/ReadVariableOp?8multi_head_attention_3/dense_21/Tensordot/ReadVariableOp?6multi_head_attention_3/dense_22/BiasAdd/ReadVariableOp?8multi_head_attention_3/dense_22/Tensordot/ReadVariableOp?6multi_head_attention_3/dense_23/BiasAdd/ReadVariableOp?8multi_head_attention_3/dense_23/Tensordot/ReadVariableOp?,sequential_3/dense_24/BiasAdd/ReadVariableOp?.sequential_3/dense_24/Tensordot/ReadVariableOp?,sequential_3/dense_25/BiasAdd/ReadVariableOp?.sequential_3/dense_25/Tensordot/ReadVariableOpR
multi_head_attention_3/ShapeShapeinputs*
T0*
_output_shapes
:t
*multi_head_attention_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,multi_head_attention_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,multi_head_attention_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$multi_head_attention_3/strided_sliceStridedSlice%multi_head_attention_3/Shape:output:03multi_head_attention_3/strided_slice/stack:output:05multi_head_attention_3/strided_slice/stack_1:output:05multi_head_attention_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
8multi_head_attention_3/dense_20/Tensordot/ReadVariableOpReadVariableOpAmulti_head_attention_3_dense_20_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0x
.multi_head_attention_3/dense_20/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
.multi_head_attention_3/dense_20/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       e
/multi_head_attention_3/dense_20/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:y
7multi_head_attention_3/dense_20/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_attention_3/dense_20/Tensordot/GatherV2GatherV28multi_head_attention_3/dense_20/Tensordot/Shape:output:07multi_head_attention_3/dense_20/Tensordot/free:output:0@multi_head_attention_3/dense_20/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9multi_head_attention_3/dense_20/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_attention_3/dense_20/Tensordot/GatherV2_1GatherV28multi_head_attention_3/dense_20/Tensordot/Shape:output:07multi_head_attention_3/dense_20/Tensordot/axes:output:0Bmulti_head_attention_3/dense_20/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/multi_head_attention_3/dense_20/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.multi_head_attention_3/dense_20/Tensordot/ProdProd;multi_head_attention_3/dense_20/Tensordot/GatherV2:output:08multi_head_attention_3/dense_20/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_attention_3/dense_20/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
0multi_head_attention_3/dense_20/Tensordot/Prod_1Prod=multi_head_attention_3/dense_20/Tensordot/GatherV2_1:output:0:multi_head_attention_3/dense_20/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5multi_head_attention_3/dense_20/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0multi_head_attention_3/dense_20/Tensordot/concatConcatV27multi_head_attention_3/dense_20/Tensordot/free:output:07multi_head_attention_3/dense_20/Tensordot/axes:output:0>multi_head_attention_3/dense_20/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
/multi_head_attention_3/dense_20/Tensordot/stackPack7multi_head_attention_3/dense_20/Tensordot/Prod:output:09multi_head_attention_3/dense_20/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
3multi_head_attention_3/dense_20/Tensordot/transpose	Transposeinputs9multi_head_attention_3/dense_20/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
1multi_head_attention_3/dense_20/Tensordot/ReshapeReshape7multi_head_attention_3/dense_20/Tensordot/transpose:y:08multi_head_attention_3/dense_20/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
0multi_head_attention_3/dense_20/Tensordot/MatMulMatMul:multi_head_attention_3/dense_20/Tensordot/Reshape:output:0@multi_head_attention_3/dense_20/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????{
1multi_head_attention_3/dense_20/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:y
7multi_head_attention_3/dense_20/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_attention_3/dense_20/Tensordot/concat_1ConcatV2;multi_head_attention_3/dense_20/Tensordot/GatherV2:output:0:multi_head_attention_3/dense_20/Tensordot/Const_2:output:0@multi_head_attention_3/dense_20/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
)multi_head_attention_3/dense_20/TensordotReshape:multi_head_attention_3/dense_20/Tensordot/MatMul:product:0;multi_head_attention_3/dense_20/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
6multi_head_attention_3/dense_20/BiasAdd/ReadVariableOpReadVariableOp?multi_head_attention_3_dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'multi_head_attention_3/dense_20/BiasAddBiasAdd2multi_head_attention_3/dense_20/Tensordot:output:0>multi_head_attention_3/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
8multi_head_attention_3/dense_21/Tensordot/ReadVariableOpReadVariableOpAmulti_head_attention_3_dense_21_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0x
.multi_head_attention_3/dense_21/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
.multi_head_attention_3/dense_21/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       e
/multi_head_attention_3/dense_21/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:y
7multi_head_attention_3/dense_21/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_attention_3/dense_21/Tensordot/GatherV2GatherV28multi_head_attention_3/dense_21/Tensordot/Shape:output:07multi_head_attention_3/dense_21/Tensordot/free:output:0@multi_head_attention_3/dense_21/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9multi_head_attention_3/dense_21/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_attention_3/dense_21/Tensordot/GatherV2_1GatherV28multi_head_attention_3/dense_21/Tensordot/Shape:output:07multi_head_attention_3/dense_21/Tensordot/axes:output:0Bmulti_head_attention_3/dense_21/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/multi_head_attention_3/dense_21/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.multi_head_attention_3/dense_21/Tensordot/ProdProd;multi_head_attention_3/dense_21/Tensordot/GatherV2:output:08multi_head_attention_3/dense_21/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_attention_3/dense_21/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
0multi_head_attention_3/dense_21/Tensordot/Prod_1Prod=multi_head_attention_3/dense_21/Tensordot/GatherV2_1:output:0:multi_head_attention_3/dense_21/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5multi_head_attention_3/dense_21/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0multi_head_attention_3/dense_21/Tensordot/concatConcatV27multi_head_attention_3/dense_21/Tensordot/free:output:07multi_head_attention_3/dense_21/Tensordot/axes:output:0>multi_head_attention_3/dense_21/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
/multi_head_attention_3/dense_21/Tensordot/stackPack7multi_head_attention_3/dense_21/Tensordot/Prod:output:09multi_head_attention_3/dense_21/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
3multi_head_attention_3/dense_21/Tensordot/transpose	Transposeinputs9multi_head_attention_3/dense_21/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
1multi_head_attention_3/dense_21/Tensordot/ReshapeReshape7multi_head_attention_3/dense_21/Tensordot/transpose:y:08multi_head_attention_3/dense_21/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
0multi_head_attention_3/dense_21/Tensordot/MatMulMatMul:multi_head_attention_3/dense_21/Tensordot/Reshape:output:0@multi_head_attention_3/dense_21/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????{
1multi_head_attention_3/dense_21/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:y
7multi_head_attention_3/dense_21/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_attention_3/dense_21/Tensordot/concat_1ConcatV2;multi_head_attention_3/dense_21/Tensordot/GatherV2:output:0:multi_head_attention_3/dense_21/Tensordot/Const_2:output:0@multi_head_attention_3/dense_21/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
)multi_head_attention_3/dense_21/TensordotReshape:multi_head_attention_3/dense_21/Tensordot/MatMul:product:0;multi_head_attention_3/dense_21/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
6multi_head_attention_3/dense_21/BiasAdd/ReadVariableOpReadVariableOp?multi_head_attention_3_dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'multi_head_attention_3/dense_21/BiasAddBiasAdd2multi_head_attention_3/dense_21/Tensordot:output:0>multi_head_attention_3/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
8multi_head_attention_3/dense_22/Tensordot/ReadVariableOpReadVariableOpAmulti_head_attention_3_dense_22_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0x
.multi_head_attention_3/dense_22/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
.multi_head_attention_3/dense_22/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       e
/multi_head_attention_3/dense_22/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:y
7multi_head_attention_3/dense_22/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_attention_3/dense_22/Tensordot/GatherV2GatherV28multi_head_attention_3/dense_22/Tensordot/Shape:output:07multi_head_attention_3/dense_22/Tensordot/free:output:0@multi_head_attention_3/dense_22/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9multi_head_attention_3/dense_22/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_attention_3/dense_22/Tensordot/GatherV2_1GatherV28multi_head_attention_3/dense_22/Tensordot/Shape:output:07multi_head_attention_3/dense_22/Tensordot/axes:output:0Bmulti_head_attention_3/dense_22/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/multi_head_attention_3/dense_22/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.multi_head_attention_3/dense_22/Tensordot/ProdProd;multi_head_attention_3/dense_22/Tensordot/GatherV2:output:08multi_head_attention_3/dense_22/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_attention_3/dense_22/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
0multi_head_attention_3/dense_22/Tensordot/Prod_1Prod=multi_head_attention_3/dense_22/Tensordot/GatherV2_1:output:0:multi_head_attention_3/dense_22/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5multi_head_attention_3/dense_22/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0multi_head_attention_3/dense_22/Tensordot/concatConcatV27multi_head_attention_3/dense_22/Tensordot/free:output:07multi_head_attention_3/dense_22/Tensordot/axes:output:0>multi_head_attention_3/dense_22/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
/multi_head_attention_3/dense_22/Tensordot/stackPack7multi_head_attention_3/dense_22/Tensordot/Prod:output:09multi_head_attention_3/dense_22/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
3multi_head_attention_3/dense_22/Tensordot/transpose	Transposeinputs9multi_head_attention_3/dense_22/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
1multi_head_attention_3/dense_22/Tensordot/ReshapeReshape7multi_head_attention_3/dense_22/Tensordot/transpose:y:08multi_head_attention_3/dense_22/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
0multi_head_attention_3/dense_22/Tensordot/MatMulMatMul:multi_head_attention_3/dense_22/Tensordot/Reshape:output:0@multi_head_attention_3/dense_22/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????{
1multi_head_attention_3/dense_22/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:y
7multi_head_attention_3/dense_22/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_attention_3/dense_22/Tensordot/concat_1ConcatV2;multi_head_attention_3/dense_22/Tensordot/GatherV2:output:0:multi_head_attention_3/dense_22/Tensordot/Const_2:output:0@multi_head_attention_3/dense_22/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
)multi_head_attention_3/dense_22/TensordotReshape:multi_head_attention_3/dense_22/Tensordot/MatMul:product:0;multi_head_attention_3/dense_22/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
6multi_head_attention_3/dense_22/BiasAdd/ReadVariableOpReadVariableOp?multi_head_attention_3_dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'multi_head_attention_3/dense_22/BiasAddBiasAdd2multi_head_attention_3/dense_22/Tensordot:output:0>multi_head_attention_3/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'q
&multi_head_attention_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????h
&multi_head_attention_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :h
&multi_head_attention_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
$multi_head_attention_3/Reshape/shapePack-multi_head_attention_3/strided_slice:output:0/multi_head_attention_3/Reshape/shape/1:output:0/multi_head_attention_3/Reshape/shape/2:output:0/multi_head_attention_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
multi_head_attention_3/ReshapeReshape0multi_head_attention_3/dense_20/BiasAdd:output:0-multi_head_attention_3/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????~
%multi_head_attention_3/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
 multi_head_attention_3/transpose	Transpose'multi_head_attention_3/Reshape:output:0.multi_head_attention_3/transpose/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????s
(multi_head_attention_3/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????j
(multi_head_attention_3/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :j
(multi_head_attention_3/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
&multi_head_attention_3/Reshape_1/shapePack-multi_head_attention_3/strided_slice:output:01multi_head_attention_3/Reshape_1/shape/1:output:01multi_head_attention_3/Reshape_1/shape/2:output:01multi_head_attention_3/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:?
 multi_head_attention_3/Reshape_1Reshape0multi_head_attention_3/dense_21/BiasAdd:output:0/multi_head_attention_3/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"???????????????????
'multi_head_attention_3/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
"multi_head_attention_3/transpose_1	Transpose)multi_head_attention_3/Reshape_1:output:00multi_head_attention_3/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????s
(multi_head_attention_3/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????j
(multi_head_attention_3/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :j
(multi_head_attention_3/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
&multi_head_attention_3/Reshape_2/shapePack-multi_head_attention_3/strided_slice:output:01multi_head_attention_3/Reshape_2/shape/1:output:01multi_head_attention_3/Reshape_2/shape/2:output:01multi_head_attention_3/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:?
 multi_head_attention_3/Reshape_2Reshape0multi_head_attention_3/dense_22/BiasAdd:output:0/multi_head_attention_3/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"???????????????????
'multi_head_attention_3/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
"multi_head_attention_3/transpose_2	Transpose)multi_head_attention_3/Reshape_2:output:00multi_head_attention_3/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"???????????????????
multi_head_attention_3/MatMulBatchMatMulV2$multi_head_attention_3/transpose:y:0&multi_head_attention_3/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(t
multi_head_attention_3/Shape_1Shape&multi_head_attention_3/transpose_1:y:0*
T0*
_output_shapes
:
,multi_head_attention_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????x
.multi_head_attention_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.multi_head_attention_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&multi_head_attention_3/strided_slice_1StridedSlice'multi_head_attention_3/Shape_1:output:05multi_head_attention_3/strided_slice_1/stack:output:07multi_head_attention_3/strided_slice_1/stack_1:output:07multi_head_attention_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
multi_head_attention_3/CastCast/multi_head_attention_3/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: e
multi_head_attention_3/SqrtSqrtmulti_head_attention_3/Cast:y:0*
T0*
_output_shapes
: ?
multi_head_attention_3/truedivRealDiv&multi_head_attention_3/MatMul:output:0multi_head_attention_3/Sqrt:y:0*
T0*A
_output_shapes/
-:+????????????????????????????
multi_head_attention_3/SoftmaxSoftmax"multi_head_attention_3/truediv:z:0*
T0*A
_output_shapes/
-:+????????????????????????????
multi_head_attention_3/MatMul_1BatchMatMulV2(multi_head_attention_3/Softmax:softmax:0&multi_head_attention_3/transpose_2:y:0*
T0*8
_output_shapes&
$:"???????????????????
'multi_head_attention_3/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
"multi_head_attention_3/transpose_3	Transpose(multi_head_attention_3/MatMul_1:output:00multi_head_attention_3/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????s
(multi_head_attention_3/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????j
(multi_head_attention_3/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
&multi_head_attention_3/Reshape_3/shapePack-multi_head_attention_3/strided_slice:output:01multi_head_attention_3/Reshape_3/shape/1:output:01multi_head_attention_3/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:?
 multi_head_attention_3/Reshape_3Reshape&multi_head_attention_3/transpose_3:y:0/multi_head_attention_3/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :???????????????????
8multi_head_attention_3/dense_23/Tensordot/ReadVariableOpReadVariableOpAmulti_head_attention_3_dense_23_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0x
.multi_head_attention_3/dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
.multi_head_attention_3/dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
/multi_head_attention_3/dense_23/Tensordot/ShapeShape)multi_head_attention_3/Reshape_3:output:0*
T0*
_output_shapes
:y
7multi_head_attention_3/dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_attention_3/dense_23/Tensordot/GatherV2GatherV28multi_head_attention_3/dense_23/Tensordot/Shape:output:07multi_head_attention_3/dense_23/Tensordot/free:output:0@multi_head_attention_3/dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9multi_head_attention_3/dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_attention_3/dense_23/Tensordot/GatherV2_1GatherV28multi_head_attention_3/dense_23/Tensordot/Shape:output:07multi_head_attention_3/dense_23/Tensordot/axes:output:0Bmulti_head_attention_3/dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/multi_head_attention_3/dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.multi_head_attention_3/dense_23/Tensordot/ProdProd;multi_head_attention_3/dense_23/Tensordot/GatherV2:output:08multi_head_attention_3/dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_attention_3/dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
0multi_head_attention_3/dense_23/Tensordot/Prod_1Prod=multi_head_attention_3/dense_23/Tensordot/GatherV2_1:output:0:multi_head_attention_3/dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5multi_head_attention_3/dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0multi_head_attention_3/dense_23/Tensordot/concatConcatV27multi_head_attention_3/dense_23/Tensordot/free:output:07multi_head_attention_3/dense_23/Tensordot/axes:output:0>multi_head_attention_3/dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
/multi_head_attention_3/dense_23/Tensordot/stackPack7multi_head_attention_3/dense_23/Tensordot/Prod:output:09multi_head_attention_3/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
3multi_head_attention_3/dense_23/Tensordot/transpose	Transpose)multi_head_attention_3/Reshape_3:output:09multi_head_attention_3/dense_23/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :???????????????????
1multi_head_attention_3/dense_23/Tensordot/ReshapeReshape7multi_head_attention_3/dense_23/Tensordot/transpose:y:08multi_head_attention_3/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
0multi_head_attention_3/dense_23/Tensordot/MatMulMatMul:multi_head_attention_3/dense_23/Tensordot/Reshape:output:0@multi_head_attention_3/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????{
1multi_head_attention_3/dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:y
7multi_head_attention_3/dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_attention_3/dense_23/Tensordot/concat_1ConcatV2;multi_head_attention_3/dense_23/Tensordot/GatherV2:output:0:multi_head_attention_3/dense_23/Tensordot/Const_2:output:0@multi_head_attention_3/dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
)multi_head_attention_3/dense_23/TensordotReshape:multi_head_attention_3/dense_23/Tensordot/MatMul:product:0;multi_head_attention_3/dense_23/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :???????????????????
6multi_head_attention_3/dense_23/BiasAdd/ReadVariableOpReadVariableOp?multi_head_attention_3_dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'multi_head_attention_3/dense_23/BiasAddBiasAdd2multi_head_attention_3/dense_23/Tensordot:output:0>multi_head_attention_3/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :???????????????????
dropout_8/IdentityIdentity0multi_head_attention_3/dense_23/BiasAdd:output:0*
T0*4
_output_shapes"
 :??????????????????h
addAddV2inputsdropout_8/Identity:output:0*
T0*,
_output_shapes
:??????????'~
4layer_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
"layer_normalization_6/moments/meanMeanadd:z:0=layer_normalization_6/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????'*
	keep_dims(?
*layer_normalization_6/moments/StopGradientStopGradient+layer_normalization_6/moments/mean:output:0*
T0*,
_output_shapes
:??????????'?
/layer_normalization_6/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_6/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????'?
8layer_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_6/moments/varianceMean3layer_normalization_6/moments/SquaredDifference:z:0Alayer_normalization_6/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????'*
	keep_dims(j
%layer_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?5?
#layer_normalization_6/batchnorm/addAddV2/layer_normalization_6/moments/variance:output:0.layer_normalization_6/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????'?
%layer_normalization_6/batchnorm/RsqrtRsqrt'layer_normalization_6/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????'?
2layer_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
#layer_normalization_6/batchnorm/mulMul)layer_normalization_6/batchnorm/Rsqrt:y:0:layer_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
%layer_normalization_6/batchnorm/mul_1Muladd:z:0'layer_normalization_6/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????'?
%layer_normalization_6/batchnorm/mul_2Mul+layer_normalization_6/moments/mean:output:0'layer_normalization_6/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????'?
.layer_normalization_6/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
#layer_normalization_6/batchnorm/subSub6layer_normalization_6/batchnorm/ReadVariableOp:value:0)layer_normalization_6/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????'?
%layer_normalization_6/batchnorm/add_1AddV2)layer_normalization_6/batchnorm/mul_1:z:0'layer_normalization_6/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????'?
.sequential_3/dense_24/Tensordot/ReadVariableOpReadVariableOp7sequential_3_dense_24_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0n
$sequential_3/dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:u
$sequential_3/dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ~
%sequential_3/dense_24/Tensordot/ShapeShape)layer_normalization_6/batchnorm/add_1:z:0*
T0*
_output_shapes
:o
-sequential_3/dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(sequential_3/dense_24/Tensordot/GatherV2GatherV2.sequential_3/dense_24/Tensordot/Shape:output:0-sequential_3/dense_24/Tensordot/free:output:06sequential_3/dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:q
/sequential_3/dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
*sequential_3/dense_24/Tensordot/GatherV2_1GatherV2.sequential_3/dense_24/Tensordot/Shape:output:0-sequential_3/dense_24/Tensordot/axes:output:08sequential_3/dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
%sequential_3/dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
$sequential_3/dense_24/Tensordot/ProdProd1sequential_3/dense_24/Tensordot/GatherV2:output:0.sequential_3/dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'sequential_3/dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
&sequential_3/dense_24/Tensordot/Prod_1Prod3sequential_3/dense_24/Tensordot/GatherV2_1:output:00sequential_3/dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+sequential_3/dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
&sequential_3/dense_24/Tensordot/concatConcatV2-sequential_3/dense_24/Tensordot/free:output:0-sequential_3/dense_24/Tensordot/axes:output:04sequential_3/dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
%sequential_3/dense_24/Tensordot/stackPack-sequential_3/dense_24/Tensordot/Prod:output:0/sequential_3/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
)sequential_3/dense_24/Tensordot/transpose	Transpose)layer_normalization_6/batchnorm/add_1:z:0/sequential_3/dense_24/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
'sequential_3/dense_24/Tensordot/ReshapeReshape-sequential_3/dense_24/Tensordot/transpose:y:0.sequential_3/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
&sequential_3/dense_24/Tensordot/MatMulMatMul0sequential_3/dense_24/Tensordot/Reshape:output:06sequential_3/dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????q
'sequential_3/dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:o
-sequential_3/dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(sequential_3/dense_24/Tensordot/concat_1ConcatV21sequential_3/dense_24/Tensordot/GatherV2:output:00sequential_3/dense_24/Tensordot/Const_2:output:06sequential_3/dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
sequential_3/dense_24/TensordotReshape0sequential_3/dense_24/Tensordot/MatMul:product:01sequential_3/dense_24/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
,sequential_3/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_3/dense_24/BiasAddBiasAdd(sequential_3/dense_24/Tensordot:output:04sequential_3/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
sequential_3/dense_24/ReluRelu&sequential_3/dense_24/BiasAdd:output:0*
T0*,
_output_shapes
:??????????'?
.sequential_3/dense_25/Tensordot/ReadVariableOpReadVariableOp7sequential_3_dense_25_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0n
$sequential_3/dense_25/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:u
$sequential_3/dense_25/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
%sequential_3/dense_25/Tensordot/ShapeShape(sequential_3/dense_24/Relu:activations:0*
T0*
_output_shapes
:o
-sequential_3/dense_25/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(sequential_3/dense_25/Tensordot/GatherV2GatherV2.sequential_3/dense_25/Tensordot/Shape:output:0-sequential_3/dense_25/Tensordot/free:output:06sequential_3/dense_25/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:q
/sequential_3/dense_25/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
*sequential_3/dense_25/Tensordot/GatherV2_1GatherV2.sequential_3/dense_25/Tensordot/Shape:output:0-sequential_3/dense_25/Tensordot/axes:output:08sequential_3/dense_25/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
%sequential_3/dense_25/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
$sequential_3/dense_25/Tensordot/ProdProd1sequential_3/dense_25/Tensordot/GatherV2:output:0.sequential_3/dense_25/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'sequential_3/dense_25/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
&sequential_3/dense_25/Tensordot/Prod_1Prod3sequential_3/dense_25/Tensordot/GatherV2_1:output:00sequential_3/dense_25/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+sequential_3/dense_25/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
&sequential_3/dense_25/Tensordot/concatConcatV2-sequential_3/dense_25/Tensordot/free:output:0-sequential_3/dense_25/Tensordot/axes:output:04sequential_3/dense_25/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
%sequential_3/dense_25/Tensordot/stackPack-sequential_3/dense_25/Tensordot/Prod:output:0/sequential_3/dense_25/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
)sequential_3/dense_25/Tensordot/transpose	Transpose(sequential_3/dense_24/Relu:activations:0/sequential_3/dense_25/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
'sequential_3/dense_25/Tensordot/ReshapeReshape-sequential_3/dense_25/Tensordot/transpose:y:0.sequential_3/dense_25/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
&sequential_3/dense_25/Tensordot/MatMulMatMul0sequential_3/dense_25/Tensordot/Reshape:output:06sequential_3/dense_25/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????q
'sequential_3/dense_25/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:o
-sequential_3/dense_25/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(sequential_3/dense_25/Tensordot/concat_1ConcatV21sequential_3/dense_25/Tensordot/GatherV2:output:00sequential_3/dense_25/Tensordot/Const_2:output:06sequential_3/dense_25/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
sequential_3/dense_25/TensordotReshape0sequential_3/dense_25/Tensordot/MatMul:product:01sequential_3/dense_25/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
,sequential_3/dense_25/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_3/dense_25/BiasAddBiasAdd(sequential_3/dense_25/Tensordot:output:04sequential_3/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'}
dropout_9/IdentityIdentity&sequential_3/dense_25/BiasAdd:output:0*
T0*,
_output_shapes
:??????????'?
add_1AddV2)layer_normalization_6/batchnorm/add_1:z:0dropout_9/Identity:output:0*
T0*,
_output_shapes
:??????????'~
4layer_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
"layer_normalization_7/moments/meanMean	add_1:z:0=layer_normalization_7/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????'*
	keep_dims(?
*layer_normalization_7/moments/StopGradientStopGradient+layer_normalization_7/moments/mean:output:0*
T0*,
_output_shapes
:??????????'?
/layer_normalization_7/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_7/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????'?
8layer_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_7/moments/varianceMean3layer_normalization_7/moments/SquaredDifference:z:0Alayer_normalization_7/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????'*
	keep_dims(j
%layer_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?5?
#layer_normalization_7/batchnorm/addAddV2/layer_normalization_7/moments/variance:output:0.layer_normalization_7/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????'?
%layer_normalization_7/batchnorm/RsqrtRsqrt'layer_normalization_7/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????'?
2layer_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
#layer_normalization_7/batchnorm/mulMul)layer_normalization_7/batchnorm/Rsqrt:y:0:layer_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
%layer_normalization_7/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_7/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????'?
%layer_normalization_7/batchnorm/mul_2Mul+layer_normalization_7/moments/mean:output:0'layer_normalization_7/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????'?
.layer_normalization_7/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
#layer_normalization_7/batchnorm/subSub6layer_normalization_7/batchnorm/ReadVariableOp:value:0)layer_normalization_7/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????'?
%layer_normalization_7/batchnorm/add_1AddV2)layer_normalization_7/batchnorm/mul_1:z:0'layer_normalization_7/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????'}
IdentityIdentity)layer_normalization_7/batchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:??????????'?
NoOpNoOp/^layer_normalization_6/batchnorm/ReadVariableOp3^layer_normalization_6/batchnorm/mul/ReadVariableOp/^layer_normalization_7/batchnorm/ReadVariableOp3^layer_normalization_7/batchnorm/mul/ReadVariableOp7^multi_head_attention_3/dense_20/BiasAdd/ReadVariableOp9^multi_head_attention_3/dense_20/Tensordot/ReadVariableOp7^multi_head_attention_3/dense_21/BiasAdd/ReadVariableOp9^multi_head_attention_3/dense_21/Tensordot/ReadVariableOp7^multi_head_attention_3/dense_22/BiasAdd/ReadVariableOp9^multi_head_attention_3/dense_22/Tensordot/ReadVariableOp7^multi_head_attention_3/dense_23/BiasAdd/ReadVariableOp9^multi_head_attention_3/dense_23/Tensordot/ReadVariableOp-^sequential_3/dense_24/BiasAdd/ReadVariableOp/^sequential_3/dense_24/Tensordot/ReadVariableOp-^sequential_3/dense_25/BiasAdd/ReadVariableOp/^sequential_3/dense_25/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:??????????': : : : : : : : : : : : : : : : 2`
.layer_normalization_6/batchnorm/ReadVariableOp.layer_normalization_6/batchnorm/ReadVariableOp2h
2layer_normalization_6/batchnorm/mul/ReadVariableOp2layer_normalization_6/batchnorm/mul/ReadVariableOp2`
.layer_normalization_7/batchnorm/ReadVariableOp.layer_normalization_7/batchnorm/ReadVariableOp2h
2layer_normalization_7/batchnorm/mul/ReadVariableOp2layer_normalization_7/batchnorm/mul/ReadVariableOp2p
6multi_head_attention_3/dense_20/BiasAdd/ReadVariableOp6multi_head_attention_3/dense_20/BiasAdd/ReadVariableOp2t
8multi_head_attention_3/dense_20/Tensordot/ReadVariableOp8multi_head_attention_3/dense_20/Tensordot/ReadVariableOp2p
6multi_head_attention_3/dense_21/BiasAdd/ReadVariableOp6multi_head_attention_3/dense_21/BiasAdd/ReadVariableOp2t
8multi_head_attention_3/dense_21/Tensordot/ReadVariableOp8multi_head_attention_3/dense_21/Tensordot/ReadVariableOp2p
6multi_head_attention_3/dense_22/BiasAdd/ReadVariableOp6multi_head_attention_3/dense_22/BiasAdd/ReadVariableOp2t
8multi_head_attention_3/dense_22/Tensordot/ReadVariableOp8multi_head_attention_3/dense_22/Tensordot/ReadVariableOp2p
6multi_head_attention_3/dense_23/BiasAdd/ReadVariableOp6multi_head_attention_3/dense_23/BiasAdd/ReadVariableOp2t
8multi_head_attention_3/dense_23/Tensordot/ReadVariableOp8multi_head_attention_3/dense_23/Tensordot/ReadVariableOp2\
,sequential_3/dense_24/BiasAdd/ReadVariableOp,sequential_3/dense_24/BiasAdd/ReadVariableOp2`
.sequential_3/dense_24/Tensordot/ReadVariableOp.sequential_3/dense_24/Tensordot/ReadVariableOp2\
,sequential_3/dense_25/BiasAdd/ReadVariableOp,sequential_3/dense_25/BiasAdd/ReadVariableOp2`
.sequential_3/dense_25/Tensordot/ReadVariableOp.sequential_3/dense_25/Tensordot/ReadVariableOp:T P
,
_output_shapes
:??????????'
 
_user_specified_nameinputs
?
?
,__inference_sequential_3_layer_call_fn_16935
dense_24_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_24_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????'*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_16911t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????'`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????': : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
,
_output_shapes
:??????????'
(
_user_specified_namedense_24_input
??
?*
__inference__traced_save_20069
file_prefix.
*savev2_dense_26_kernel_read_readvariableop,
(savev2_dense_26_bias_read_readvariableop.
*savev2_dense_27_kernel_read_readvariableop,
(savev2_dense_27_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopV
Rsavev2_token_and_position_embedding_14_embedding_32_embeddings_read_readvariableopV
Rsavev2_token_and_position_embedding_14_embedding_33_embeddings_read_readvariableopY
Usavev2_transformer_block_3_multi_head_attention_3_dense_20_kernel_read_readvariableopW
Ssavev2_transformer_block_3_multi_head_attention_3_dense_20_bias_read_readvariableopY
Usavev2_transformer_block_3_multi_head_attention_3_dense_21_kernel_read_readvariableopW
Ssavev2_transformer_block_3_multi_head_attention_3_dense_21_bias_read_readvariableopY
Usavev2_transformer_block_3_multi_head_attention_3_dense_22_kernel_read_readvariableopW
Ssavev2_transformer_block_3_multi_head_attention_3_dense_22_bias_read_readvariableopY
Usavev2_transformer_block_3_multi_head_attention_3_dense_23_kernel_read_readvariableopW
Ssavev2_transformer_block_3_multi_head_attention_3_dense_23_bias_read_readvariableop.
*savev2_dense_24_kernel_read_readvariableop,
(savev2_dense_24_bias_read_readvariableop.
*savev2_dense_25_kernel_read_readvariableop,
(savev2_dense_25_bias_read_readvariableopN
Jsavev2_transformer_block_3_layer_normalization_6_gamma_read_readvariableopM
Isavev2_transformer_block_3_layer_normalization_6_beta_read_readvariableopN
Jsavev2_transformer_block_3_layer_normalization_7_gamma_read_readvariableopM
Isavev2_transformer_block_3_layer_normalization_7_beta_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_26_kernel_m_read_readvariableop3
/savev2_adam_dense_26_bias_m_read_readvariableop5
1savev2_adam_dense_27_kernel_m_read_readvariableop3
/savev2_adam_dense_27_bias_m_read_readvariableop]
Ysavev2_adam_token_and_position_embedding_14_embedding_32_embeddings_m_read_readvariableop]
Ysavev2_adam_token_and_position_embedding_14_embedding_33_embeddings_m_read_readvariableop`
\savev2_adam_transformer_block_3_multi_head_attention_3_dense_20_kernel_m_read_readvariableop^
Zsavev2_adam_transformer_block_3_multi_head_attention_3_dense_20_bias_m_read_readvariableop`
\savev2_adam_transformer_block_3_multi_head_attention_3_dense_21_kernel_m_read_readvariableop^
Zsavev2_adam_transformer_block_3_multi_head_attention_3_dense_21_bias_m_read_readvariableop`
\savev2_adam_transformer_block_3_multi_head_attention_3_dense_22_kernel_m_read_readvariableop^
Zsavev2_adam_transformer_block_3_multi_head_attention_3_dense_22_bias_m_read_readvariableop`
\savev2_adam_transformer_block_3_multi_head_attention_3_dense_23_kernel_m_read_readvariableop^
Zsavev2_adam_transformer_block_3_multi_head_attention_3_dense_23_bias_m_read_readvariableop5
1savev2_adam_dense_24_kernel_m_read_readvariableop3
/savev2_adam_dense_24_bias_m_read_readvariableop5
1savev2_adam_dense_25_kernel_m_read_readvariableop3
/savev2_adam_dense_25_bias_m_read_readvariableopU
Qsavev2_adam_transformer_block_3_layer_normalization_6_gamma_m_read_readvariableopT
Psavev2_adam_transformer_block_3_layer_normalization_6_beta_m_read_readvariableopU
Qsavev2_adam_transformer_block_3_layer_normalization_7_gamma_m_read_readvariableopT
Psavev2_adam_transformer_block_3_layer_normalization_7_beta_m_read_readvariableop5
1savev2_adam_dense_26_kernel_v_read_readvariableop3
/savev2_adam_dense_26_bias_v_read_readvariableop5
1savev2_adam_dense_27_kernel_v_read_readvariableop3
/savev2_adam_dense_27_bias_v_read_readvariableop]
Ysavev2_adam_token_and_position_embedding_14_embedding_32_embeddings_v_read_readvariableop]
Ysavev2_adam_token_and_position_embedding_14_embedding_33_embeddings_v_read_readvariableop`
\savev2_adam_transformer_block_3_multi_head_attention_3_dense_20_kernel_v_read_readvariableop^
Zsavev2_adam_transformer_block_3_multi_head_attention_3_dense_20_bias_v_read_readvariableop`
\savev2_adam_transformer_block_3_multi_head_attention_3_dense_21_kernel_v_read_readvariableop^
Zsavev2_adam_transformer_block_3_multi_head_attention_3_dense_21_bias_v_read_readvariableop`
\savev2_adam_transformer_block_3_multi_head_attention_3_dense_22_kernel_v_read_readvariableop^
Zsavev2_adam_transformer_block_3_multi_head_attention_3_dense_22_bias_v_read_readvariableop`
\savev2_adam_transformer_block_3_multi_head_attention_3_dense_23_kernel_v_read_readvariableop^
Zsavev2_adam_transformer_block_3_multi_head_attention_3_dense_23_bias_v_read_readvariableop5
1savev2_adam_dense_24_kernel_v_read_readvariableop3
/savev2_adam_dense_24_bias_v_read_readvariableop5
1savev2_adam_dense_25_kernel_v_read_readvariableop3
/savev2_adam_dense_25_bias_v_read_readvariableopU
Qsavev2_adam_transformer_block_3_layer_normalization_6_gamma_v_read_readvariableopT
Psavev2_adam_transformer_block_3_layer_normalization_6_beta_v_read_readvariableopU
Qsavev2_adam_transformer_block_3_layer_normalization_7_gamma_v_read_readvariableopT
Psavev2_adam_transformer_block_3_layer_normalization_7_beta_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?#
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*?"
value?"B?"JB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*?
value?B?JB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?)
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_26_kernel_read_readvariableop(savev2_dense_26_bias_read_readvariableop*savev2_dense_27_kernel_read_readvariableop(savev2_dense_27_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopRsavev2_token_and_position_embedding_14_embedding_32_embeddings_read_readvariableopRsavev2_token_and_position_embedding_14_embedding_33_embeddings_read_readvariableopUsavev2_transformer_block_3_multi_head_attention_3_dense_20_kernel_read_readvariableopSsavev2_transformer_block_3_multi_head_attention_3_dense_20_bias_read_readvariableopUsavev2_transformer_block_3_multi_head_attention_3_dense_21_kernel_read_readvariableopSsavev2_transformer_block_3_multi_head_attention_3_dense_21_bias_read_readvariableopUsavev2_transformer_block_3_multi_head_attention_3_dense_22_kernel_read_readvariableopSsavev2_transformer_block_3_multi_head_attention_3_dense_22_bias_read_readvariableopUsavev2_transformer_block_3_multi_head_attention_3_dense_23_kernel_read_readvariableopSsavev2_transformer_block_3_multi_head_attention_3_dense_23_bias_read_readvariableop*savev2_dense_24_kernel_read_readvariableop(savev2_dense_24_bias_read_readvariableop*savev2_dense_25_kernel_read_readvariableop(savev2_dense_25_bias_read_readvariableopJsavev2_transformer_block_3_layer_normalization_6_gamma_read_readvariableopIsavev2_transformer_block_3_layer_normalization_6_beta_read_readvariableopJsavev2_transformer_block_3_layer_normalization_7_gamma_read_readvariableopIsavev2_transformer_block_3_layer_normalization_7_beta_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_26_kernel_m_read_readvariableop/savev2_adam_dense_26_bias_m_read_readvariableop1savev2_adam_dense_27_kernel_m_read_readvariableop/savev2_adam_dense_27_bias_m_read_readvariableopYsavev2_adam_token_and_position_embedding_14_embedding_32_embeddings_m_read_readvariableopYsavev2_adam_token_and_position_embedding_14_embedding_33_embeddings_m_read_readvariableop\savev2_adam_transformer_block_3_multi_head_attention_3_dense_20_kernel_m_read_readvariableopZsavev2_adam_transformer_block_3_multi_head_attention_3_dense_20_bias_m_read_readvariableop\savev2_adam_transformer_block_3_multi_head_attention_3_dense_21_kernel_m_read_readvariableopZsavev2_adam_transformer_block_3_multi_head_attention_3_dense_21_bias_m_read_readvariableop\savev2_adam_transformer_block_3_multi_head_attention_3_dense_22_kernel_m_read_readvariableopZsavev2_adam_transformer_block_3_multi_head_attention_3_dense_22_bias_m_read_readvariableop\savev2_adam_transformer_block_3_multi_head_attention_3_dense_23_kernel_m_read_readvariableopZsavev2_adam_transformer_block_3_multi_head_attention_3_dense_23_bias_m_read_readvariableop1savev2_adam_dense_24_kernel_m_read_readvariableop/savev2_adam_dense_24_bias_m_read_readvariableop1savev2_adam_dense_25_kernel_m_read_readvariableop/savev2_adam_dense_25_bias_m_read_readvariableopQsavev2_adam_transformer_block_3_layer_normalization_6_gamma_m_read_readvariableopPsavev2_adam_transformer_block_3_layer_normalization_6_beta_m_read_readvariableopQsavev2_adam_transformer_block_3_layer_normalization_7_gamma_m_read_readvariableopPsavev2_adam_transformer_block_3_layer_normalization_7_beta_m_read_readvariableop1savev2_adam_dense_26_kernel_v_read_readvariableop/savev2_adam_dense_26_bias_v_read_readvariableop1savev2_adam_dense_27_kernel_v_read_readvariableop/savev2_adam_dense_27_bias_v_read_readvariableopYsavev2_adam_token_and_position_embedding_14_embedding_32_embeddings_v_read_readvariableopYsavev2_adam_token_and_position_embedding_14_embedding_33_embeddings_v_read_readvariableop\savev2_adam_transformer_block_3_multi_head_attention_3_dense_20_kernel_v_read_readvariableopZsavev2_adam_transformer_block_3_multi_head_attention_3_dense_20_bias_v_read_readvariableop\savev2_adam_transformer_block_3_multi_head_attention_3_dense_21_kernel_v_read_readvariableopZsavev2_adam_transformer_block_3_multi_head_attention_3_dense_21_bias_v_read_readvariableop\savev2_adam_transformer_block_3_multi_head_attention_3_dense_22_kernel_v_read_readvariableopZsavev2_adam_transformer_block_3_multi_head_attention_3_dense_22_bias_v_read_readvariableop\savev2_adam_transformer_block_3_multi_head_attention_3_dense_23_kernel_v_read_readvariableopZsavev2_adam_transformer_block_3_multi_head_attention_3_dense_23_bias_v_read_readvariableop1savev2_adam_dense_24_kernel_v_read_readvariableop/savev2_adam_dense_24_bias_v_read_readvariableop1savev2_adam_dense_25_kernel_v_read_readvariableop/savev2_adam_dense_25_bias_v_read_readvariableopQsavev2_adam_transformer_block_3_layer_normalization_6_gamma_v_read_readvariableopPsavev2_adam_transformer_block_3_layer_normalization_6_beta_v_read_readvariableopQsavev2_adam_transformer_block_3_layer_normalization_7_gamma_v_read_readvariableopPsavev2_adam_transformer_block_3_layer_normalization_7_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *X
dtypesN
L2J	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::: : : : : :	?:	?N::::::::::::::::: : :::::	?:	?N:::::::::::::::::::::	?:	?N::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :%
!

_output_shapes
:	?:%!

_output_shapes
:	?N:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$  

_output_shapes

:: !

_output_shapes
::%"!

_output_shapes
:	?:%#!

_output_shapes
:	?N:$$ 

_output_shapes

:: %

_output_shapes
::$& 

_output_shapes

:: '

_output_shapes
::$( 

_output_shapes

:: )

_output_shapes
::$* 

_output_shapes

:: +

_output_shapes
::$, 

_output_shapes

:: -

_output_shapes
::$. 

_output_shapes

:: /

_output_shapes
:: 0

_output_shapes
:: 1

_output_shapes
:: 2

_output_shapes
:: 3

_output_shapes
::$4 

_output_shapes

:: 5

_output_shapes
::$6 

_output_shapes

:: 7

_output_shapes
::%8!

_output_shapes
:	?:%9!

_output_shapes
:	?N:$: 

_output_shapes

:: ;

_output_shapes
::$< 

_output_shapes

:: =

_output_shapes
::$> 

_output_shapes

:: ?

_output_shapes
::$@ 

_output_shapes

:: A

_output_shapes
::$B 

_output_shapes

:: C

_output_shapes
::$D 

_output_shapes

:: E

_output_shapes
:: F

_output_shapes
:: G

_output_shapes
:: H

_output_shapes
:: I

_output_shapes
::J

_output_shapes
: 
?
?
,__inference_sequential_3_layer_call_fn_19621

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????'*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_16851t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????'`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????': : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????'
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_16770
input_17^
Kmodel_1_token_and_position_embedding_14_embedding_33_embedding_lookup_16500:	?N^
Kmodel_1_token_and_position_embedding_14_embedding_32_embedding_lookup_16506:	?o
]model_1_transformer_block_3_multi_head_attention_3_dense_20_tensordot_readvariableop_resource:i
[model_1_transformer_block_3_multi_head_attention_3_dense_20_biasadd_readvariableop_resource:o
]model_1_transformer_block_3_multi_head_attention_3_dense_21_tensordot_readvariableop_resource:i
[model_1_transformer_block_3_multi_head_attention_3_dense_21_biasadd_readvariableop_resource:o
]model_1_transformer_block_3_multi_head_attention_3_dense_22_tensordot_readvariableop_resource:i
[model_1_transformer_block_3_multi_head_attention_3_dense_22_biasadd_readvariableop_resource:o
]model_1_transformer_block_3_multi_head_attention_3_dense_23_tensordot_readvariableop_resource:i
[model_1_transformer_block_3_multi_head_attention_3_dense_23_biasadd_readvariableop_resource:e
Wmodel_1_transformer_block_3_layer_normalization_6_batchnorm_mul_readvariableop_resource:a
Smodel_1_transformer_block_3_layer_normalization_6_batchnorm_readvariableop_resource:e
Smodel_1_transformer_block_3_sequential_3_dense_24_tensordot_readvariableop_resource:_
Qmodel_1_transformer_block_3_sequential_3_dense_24_biasadd_readvariableop_resource:e
Smodel_1_transformer_block_3_sequential_3_dense_25_tensordot_readvariableop_resource:_
Qmodel_1_transformer_block_3_sequential_3_dense_25_biasadd_readvariableop_resource:e
Wmodel_1_transformer_block_3_layer_normalization_7_batchnorm_mul_readvariableop_resource:a
Smodel_1_transformer_block_3_layer_normalization_7_batchnorm_readvariableop_resource:A
/model_1_dense_26_matmul_readvariableop_resource:>
0model_1_dense_26_biasadd_readvariableop_resource:A
/model_1_dense_27_matmul_readvariableop_resource:>
0model_1_dense_27_biasadd_readvariableop_resource:
identity??'model_1/dense_26/BiasAdd/ReadVariableOp?&model_1/dense_26/MatMul/ReadVariableOp?'model_1/dense_27/BiasAdd/ReadVariableOp?&model_1/dense_27/MatMul/ReadVariableOp?Emodel_1/token_and_position_embedding_14/embedding_32/embedding_lookup?Emodel_1/token_and_position_embedding_14/embedding_33/embedding_lookup?Jmodel_1/transformer_block_3/layer_normalization_6/batchnorm/ReadVariableOp?Nmodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOp?Jmodel_1/transformer_block_3/layer_normalization_7/batchnorm/ReadVariableOp?Nmodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOp?Rmodel_1/transformer_block_3/multi_head_attention_3/dense_20/BiasAdd/ReadVariableOp?Tmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/ReadVariableOp?Rmodel_1/transformer_block_3/multi_head_attention_3/dense_21/BiasAdd/ReadVariableOp?Tmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/ReadVariableOp?Rmodel_1/transformer_block_3/multi_head_attention_3/dense_22/BiasAdd/ReadVariableOp?Tmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/ReadVariableOp?Rmodel_1/transformer_block_3/multi_head_attention_3/dense_23/BiasAdd/ReadVariableOp?Tmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/ReadVariableOp?Hmodel_1/transformer_block_3/sequential_3/dense_24/BiasAdd/ReadVariableOp?Jmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/ReadVariableOp?Hmodel_1/transformer_block_3/sequential_3/dense_25/BiasAdd/ReadVariableOp?Jmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/ReadVariableOp?
;model_1/token_and_position_embedding_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            ?
=model_1/token_and_position_embedding_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           ?
=model_1/token_and_position_embedding_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
5model_1/token_and_position_embedding_14/strided_sliceStridedSliceinput_17Dmodel_1/token_and_position_embedding_14/strided_slice/stack:output:0Fmodel_1/token_and_position_embedding_14/strided_slice/stack_1:output:0Fmodel_1/token_and_position_embedding_14/strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:??????????'*

begin_mask*
end_mask?
=model_1/token_and_position_embedding_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"           ?
?model_1/token_and_position_embedding_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           ?
?model_1/token_and_position_embedding_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
7model_1/token_and_position_embedding_14/strided_slice_1StridedSliceinput_17Fmodel_1/token_and_position_embedding_14/strided_slice_1/stack:output:0Hmodel_1/token_and_position_embedding_14/strided_slice_1/stack_1:output:0Hmodel_1/token_and_position_embedding_14/strided_slice_1/stack_2:output:0*
Index0*
T0*,
_output_shapes
:??????????'*

begin_mask*
end_mask?
5model_1/token_and_position_embedding_14/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
/model_1/token_and_position_embedding_14/ReshapeReshape>model_1/token_and_position_embedding_14/strided_slice:output:0>model_1/token_and_position_embedding_14/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????'?
7model_1/token_and_position_embedding_14/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
1model_1/token_and_position_embedding_14/Reshape_1Reshape@model_1/token_and_position_embedding_14/strided_slice_1:output:0@model_1/token_and_position_embedding_14/Reshape_1/shape:output:0*
T0*(
_output_shapes
:??????????'?
9model_1/token_and_position_embedding_14/embedding_33/CastCast:model_1/token_and_position_embedding_14/Reshape_1:output:0*

DstT0*

SrcT0*(
_output_shapes
:??????????'?
Emodel_1/token_and_position_embedding_14/embedding_33/embedding_lookupResourceGatherKmodel_1_token_and_position_embedding_14_embedding_33_embedding_lookup_16500=model_1/token_and_position_embedding_14/embedding_33/Cast:y:0*
Tindices0*^
_classT
RPloc:@model_1/token_and_position_embedding_14/embedding_33/embedding_lookup/16500*,
_output_shapes
:??????????'*
dtype0?
Nmodel_1/token_and_position_embedding_14/embedding_33/embedding_lookup/IdentityIdentityNmodel_1/token_and_position_embedding_14/embedding_33/embedding_lookup:output:0*
T0*^
_classT
RPloc:@model_1/token_and_position_embedding_14/embedding_33/embedding_lookup/16500*,
_output_shapes
:??????????'?
Pmodel_1/token_and_position_embedding_14/embedding_33/embedding_lookup/Identity_1IdentityWmodel_1/token_and_position_embedding_14/embedding_33/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????'?
9model_1/token_and_position_embedding_14/embedding_32/CastCast8model_1/token_and_position_embedding_14/Reshape:output:0*

DstT0*

SrcT0*(
_output_shapes
:??????????'?
Emodel_1/token_and_position_embedding_14/embedding_32/embedding_lookupResourceGatherKmodel_1_token_and_position_embedding_14_embedding_32_embedding_lookup_16506=model_1/token_and_position_embedding_14/embedding_32/Cast:y:0*
Tindices0*^
_classT
RPloc:@model_1/token_and_position_embedding_14/embedding_32/embedding_lookup/16506*,
_output_shapes
:??????????'*
dtype0?
Nmodel_1/token_and_position_embedding_14/embedding_32/embedding_lookup/IdentityIdentityNmodel_1/token_and_position_embedding_14/embedding_32/embedding_lookup:output:0*
T0*^
_classT
RPloc:@model_1/token_and_position_embedding_14/embedding_32/embedding_lookup/16506*,
_output_shapes
:??????????'?
Pmodel_1/token_and_position_embedding_14/embedding_32/embedding_lookup/Identity_1IdentityWmodel_1/token_and_position_embedding_14/embedding_32/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????'?
+model_1/token_and_position_embedding_14/addAddV2Ymodel_1/token_and_position_embedding_14/embedding_32/embedding_lookup/Identity_1:output:0Ymodel_1/token_and_position_embedding_14/embedding_33/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????'?
8model_1/transformer_block_3/multi_head_attention_3/ShapeShape/model_1/token_and_position_embedding_14/add:z:0*
T0*
_output_shapes
:?
Fmodel_1/transformer_block_3/multi_head_attention_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Hmodel_1/transformer_block_3/multi_head_attention_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Hmodel_1/transformer_block_3/multi_head_attention_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
@model_1/transformer_block_3/multi_head_attention_3/strided_sliceStridedSliceAmodel_1/transformer_block_3/multi_head_attention_3/Shape:output:0Omodel_1/transformer_block_3/multi_head_attention_3/strided_slice/stack:output:0Qmodel_1/transformer_block_3/multi_head_attention_3/strided_slice/stack_1:output:0Qmodel_1/transformer_block_3/multi_head_attention_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Tmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/ReadVariableOpReadVariableOp]model_1_transformer_block_3_multi_head_attention_3_dense_20_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0?
Jmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
Jmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
Kmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/ShapeShape/model_1/token_and_position_embedding_14/add:z:0*
T0*
_output_shapes
:?
Smodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Nmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/GatherV2GatherV2Tmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/Shape:output:0Smodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/free:output:0\model_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Umodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Pmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/GatherV2_1GatherV2Tmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/Shape:output:0Smodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/axes:output:0^model_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Kmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Jmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/ProdProdWmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/GatherV2:output:0Tmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
Mmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Lmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/Prod_1ProdYmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/GatherV2_1:output:0Vmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
Qmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Lmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/concatConcatV2Smodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/free:output:0Smodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/axes:output:0Zmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Kmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/stackPackSmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/Prod:output:0Umodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
Omodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/transpose	Transpose/model_1/token_and_position_embedding_14/add:z:0Umodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
Mmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/ReshapeReshapeSmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/transpose:y:0Tmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Lmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/MatMulMatMulVmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/Reshape:output:0\model_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Mmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Smodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Nmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/concat_1ConcatV2Wmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/GatherV2:output:0Vmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/Const_2:output:0\model_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Emodel_1/transformer_block_3/multi_head_attention_3/dense_20/TensordotReshapeVmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/MatMul:product:0Wmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
Rmodel_1/transformer_block_3/multi_head_attention_3/dense_20/BiasAdd/ReadVariableOpReadVariableOp[model_1_transformer_block_3_multi_head_attention_3_dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Cmodel_1/transformer_block_3/multi_head_attention_3/dense_20/BiasAddBiasAddNmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot:output:0Zmodel_1/transformer_block_3/multi_head_attention_3/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
Tmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/ReadVariableOpReadVariableOp]model_1_transformer_block_3_multi_head_attention_3_dense_21_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0?
Jmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
Jmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
Kmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/ShapeShape/model_1/token_and_position_embedding_14/add:z:0*
T0*
_output_shapes
:?
Smodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Nmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/GatherV2GatherV2Tmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/Shape:output:0Smodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/free:output:0\model_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Umodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Pmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/GatherV2_1GatherV2Tmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/Shape:output:0Smodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/axes:output:0^model_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Kmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Jmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/ProdProdWmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/GatherV2:output:0Tmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
Mmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Lmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/Prod_1ProdYmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/GatherV2_1:output:0Vmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
Qmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Lmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/concatConcatV2Smodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/free:output:0Smodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/axes:output:0Zmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Kmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/stackPackSmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/Prod:output:0Umodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
Omodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/transpose	Transpose/model_1/token_and_position_embedding_14/add:z:0Umodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
Mmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/ReshapeReshapeSmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/transpose:y:0Tmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Lmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/MatMulMatMulVmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/Reshape:output:0\model_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Mmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Smodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Nmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/concat_1ConcatV2Wmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/GatherV2:output:0Vmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/Const_2:output:0\model_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Emodel_1/transformer_block_3/multi_head_attention_3/dense_21/TensordotReshapeVmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/MatMul:product:0Wmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
Rmodel_1/transformer_block_3/multi_head_attention_3/dense_21/BiasAdd/ReadVariableOpReadVariableOp[model_1_transformer_block_3_multi_head_attention_3_dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Cmodel_1/transformer_block_3/multi_head_attention_3/dense_21/BiasAddBiasAddNmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot:output:0Zmodel_1/transformer_block_3/multi_head_attention_3/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
Tmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/ReadVariableOpReadVariableOp]model_1_transformer_block_3_multi_head_attention_3_dense_22_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0?
Jmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
Jmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
Kmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/ShapeShape/model_1/token_and_position_embedding_14/add:z:0*
T0*
_output_shapes
:?
Smodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Nmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/GatherV2GatherV2Tmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/Shape:output:0Smodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/free:output:0\model_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Umodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Pmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/GatherV2_1GatherV2Tmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/Shape:output:0Smodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/axes:output:0^model_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Kmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Jmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/ProdProdWmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/GatherV2:output:0Tmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
Mmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Lmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/Prod_1ProdYmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/GatherV2_1:output:0Vmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
Qmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Lmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/concatConcatV2Smodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/free:output:0Smodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/axes:output:0Zmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Kmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/stackPackSmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/Prod:output:0Umodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
Omodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/transpose	Transpose/model_1/token_and_position_embedding_14/add:z:0Umodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
Mmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/ReshapeReshapeSmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/transpose:y:0Tmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Lmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/MatMulMatMulVmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/Reshape:output:0\model_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Mmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Smodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Nmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/concat_1ConcatV2Wmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/GatherV2:output:0Vmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/Const_2:output:0\model_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Emodel_1/transformer_block_3/multi_head_attention_3/dense_22/TensordotReshapeVmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/MatMul:product:0Wmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
Rmodel_1/transformer_block_3/multi_head_attention_3/dense_22/BiasAdd/ReadVariableOpReadVariableOp[model_1_transformer_block_3_multi_head_attention_3_dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Cmodel_1/transformer_block_3/multi_head_attention_3/dense_22/BiasAddBiasAddNmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot:output:0Zmodel_1/transformer_block_3/multi_head_attention_3/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
Bmodel_1/transformer_block_3/multi_head_attention_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
??????????
Bmodel_1/transformer_block_3/multi_head_attention_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
Bmodel_1/transformer_block_3/multi_head_attention_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
@model_1/transformer_block_3/multi_head_attention_3/Reshape/shapePackImodel_1/transformer_block_3/multi_head_attention_3/strided_slice:output:0Kmodel_1/transformer_block_3/multi_head_attention_3/Reshape/shape/1:output:0Kmodel_1/transformer_block_3/multi_head_attention_3/Reshape/shape/2:output:0Kmodel_1/transformer_block_3/multi_head_attention_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
:model_1/transformer_block_3/multi_head_attention_3/ReshapeReshapeLmodel_1/transformer_block_3/multi_head_attention_3/dense_20/BiasAdd:output:0Imodel_1/transformer_block_3/multi_head_attention_3/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"???????????????????
Amodel_1/transformer_block_3/multi_head_attention_3/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
<model_1/transformer_block_3/multi_head_attention_3/transpose	TransposeCmodel_1/transformer_block_3/multi_head_attention_3/Reshape:output:0Jmodel_1/transformer_block_3/multi_head_attention_3/transpose/perm:output:0*
T0*8
_output_shapes&
$:"???????????????????
Dmodel_1/transformer_block_3/multi_head_attention_3/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
??????????
Dmodel_1/transformer_block_3/multi_head_attention_3/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
Dmodel_1/transformer_block_3/multi_head_attention_3/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
Bmodel_1/transformer_block_3/multi_head_attention_3/Reshape_1/shapePackImodel_1/transformer_block_3/multi_head_attention_3/strided_slice:output:0Mmodel_1/transformer_block_3/multi_head_attention_3/Reshape_1/shape/1:output:0Mmodel_1/transformer_block_3/multi_head_attention_3/Reshape_1/shape/2:output:0Mmodel_1/transformer_block_3/multi_head_attention_3/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:?
<model_1/transformer_block_3/multi_head_attention_3/Reshape_1ReshapeLmodel_1/transformer_block_3/multi_head_attention_3/dense_21/BiasAdd:output:0Kmodel_1/transformer_block_3/multi_head_attention_3/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"???????????????????
Cmodel_1/transformer_block_3/multi_head_attention_3/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
>model_1/transformer_block_3/multi_head_attention_3/transpose_1	TransposeEmodel_1/transformer_block_3/multi_head_attention_3/Reshape_1:output:0Lmodel_1/transformer_block_3/multi_head_attention_3/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"???????????????????
Dmodel_1/transformer_block_3/multi_head_attention_3/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
??????????
Dmodel_1/transformer_block_3/multi_head_attention_3/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
Dmodel_1/transformer_block_3/multi_head_attention_3/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
Bmodel_1/transformer_block_3/multi_head_attention_3/Reshape_2/shapePackImodel_1/transformer_block_3/multi_head_attention_3/strided_slice:output:0Mmodel_1/transformer_block_3/multi_head_attention_3/Reshape_2/shape/1:output:0Mmodel_1/transformer_block_3/multi_head_attention_3/Reshape_2/shape/2:output:0Mmodel_1/transformer_block_3/multi_head_attention_3/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:?
<model_1/transformer_block_3/multi_head_attention_3/Reshape_2ReshapeLmodel_1/transformer_block_3/multi_head_attention_3/dense_22/BiasAdd:output:0Kmodel_1/transformer_block_3/multi_head_attention_3/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"???????????????????
Cmodel_1/transformer_block_3/multi_head_attention_3/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
>model_1/transformer_block_3/multi_head_attention_3/transpose_2	TransposeEmodel_1/transformer_block_3/multi_head_attention_3/Reshape_2:output:0Lmodel_1/transformer_block_3/multi_head_attention_3/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"???????????????????
9model_1/transformer_block_3/multi_head_attention_3/MatMulBatchMatMulV2@model_1/transformer_block_3/multi_head_attention_3/transpose:y:0Bmodel_1/transformer_block_3/multi_head_attention_3/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(?
:model_1/transformer_block_3/multi_head_attention_3/Shape_1ShapeBmodel_1/transformer_block_3/multi_head_attention_3/transpose_1:y:0*
T0*
_output_shapes
:?
Hmodel_1/transformer_block_3/multi_head_attention_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Jmodel_1/transformer_block_3/multi_head_attention_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Jmodel_1/transformer_block_3/multi_head_attention_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Bmodel_1/transformer_block_3/multi_head_attention_3/strided_slice_1StridedSliceCmodel_1/transformer_block_3/multi_head_attention_3/Shape_1:output:0Qmodel_1/transformer_block_3/multi_head_attention_3/strided_slice_1/stack:output:0Smodel_1/transformer_block_3/multi_head_attention_3/strided_slice_1/stack_1:output:0Smodel_1/transformer_block_3/multi_head_attention_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
7model_1/transformer_block_3/multi_head_attention_3/CastCastKmodel_1/transformer_block_3/multi_head_attention_3/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
7model_1/transformer_block_3/multi_head_attention_3/SqrtSqrt;model_1/transformer_block_3/multi_head_attention_3/Cast:y:0*
T0*
_output_shapes
: ?
:model_1/transformer_block_3/multi_head_attention_3/truedivRealDivBmodel_1/transformer_block_3/multi_head_attention_3/MatMul:output:0;model_1/transformer_block_3/multi_head_attention_3/Sqrt:y:0*
T0*A
_output_shapes/
-:+????????????????????????????
:model_1/transformer_block_3/multi_head_attention_3/SoftmaxSoftmax>model_1/transformer_block_3/multi_head_attention_3/truediv:z:0*
T0*A
_output_shapes/
-:+????????????????????????????
;model_1/transformer_block_3/multi_head_attention_3/MatMul_1BatchMatMulV2Dmodel_1/transformer_block_3/multi_head_attention_3/Softmax:softmax:0Bmodel_1/transformer_block_3/multi_head_attention_3/transpose_2:y:0*
T0*8
_output_shapes&
$:"???????????????????
Cmodel_1/transformer_block_3/multi_head_attention_3/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
>model_1/transformer_block_3/multi_head_attention_3/transpose_3	TransposeDmodel_1/transformer_block_3/multi_head_attention_3/MatMul_1:output:0Lmodel_1/transformer_block_3/multi_head_attention_3/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"???????????????????
Dmodel_1/transformer_block_3/multi_head_attention_3/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
??????????
Dmodel_1/transformer_block_3/multi_head_attention_3/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
Bmodel_1/transformer_block_3/multi_head_attention_3/Reshape_3/shapePackImodel_1/transformer_block_3/multi_head_attention_3/strided_slice:output:0Mmodel_1/transformer_block_3/multi_head_attention_3/Reshape_3/shape/1:output:0Mmodel_1/transformer_block_3/multi_head_attention_3/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:?
<model_1/transformer_block_3/multi_head_attention_3/Reshape_3ReshapeBmodel_1/transformer_block_3/multi_head_attention_3/transpose_3:y:0Kmodel_1/transformer_block_3/multi_head_attention_3/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :???????????????????
Tmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/ReadVariableOpReadVariableOp]model_1_transformer_block_3_multi_head_attention_3_dense_23_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0?
Jmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
Jmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
Kmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/ShapeShapeEmodel_1/transformer_block_3/multi_head_attention_3/Reshape_3:output:0*
T0*
_output_shapes
:?
Smodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Nmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/GatherV2GatherV2Tmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/Shape:output:0Smodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/free:output:0\model_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Umodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Pmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/GatherV2_1GatherV2Tmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/Shape:output:0Smodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/axes:output:0^model_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Kmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Jmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/ProdProdWmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/GatherV2:output:0Tmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
Mmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Lmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/Prod_1ProdYmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/GatherV2_1:output:0Vmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
Qmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Lmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/concatConcatV2Smodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/free:output:0Smodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/axes:output:0Zmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Kmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/stackPackSmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/Prod:output:0Umodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
Omodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/transpose	TransposeEmodel_1/transformer_block_3/multi_head_attention_3/Reshape_3:output:0Umodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :???????????????????
Mmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/ReshapeReshapeSmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/transpose:y:0Tmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Lmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/MatMulMatMulVmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/Reshape:output:0\model_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Mmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Smodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Nmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/concat_1ConcatV2Wmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/GatherV2:output:0Vmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/Const_2:output:0\model_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Emodel_1/transformer_block_3/multi_head_attention_3/dense_23/TensordotReshapeVmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/MatMul:product:0Wmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :???????????????????
Rmodel_1/transformer_block_3/multi_head_attention_3/dense_23/BiasAdd/ReadVariableOpReadVariableOp[model_1_transformer_block_3_multi_head_attention_3_dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Cmodel_1/transformer_block_3/multi_head_attention_3/dense_23/BiasAddBiasAddNmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot:output:0Zmodel_1/transformer_block_3/multi_head_attention_3/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :???????????????????
.model_1/transformer_block_3/dropout_8/IdentityIdentityLmodel_1/transformer_block_3/multi_head_attention_3/dense_23/BiasAdd:output:0*
T0*4
_output_shapes"
 :???????????????????
model_1/transformer_block_3/addAddV2/model_1/token_and_position_embedding_14/add:z:07model_1/transformer_block_3/dropout_8/Identity:output:0*
T0*,
_output_shapes
:??????????'?
Pmodel_1/transformer_block_3/layer_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
>model_1/transformer_block_3/layer_normalization_6/moments/meanMean#model_1/transformer_block_3/add:z:0Ymodel_1/transformer_block_3/layer_normalization_6/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????'*
	keep_dims(?
Fmodel_1/transformer_block_3/layer_normalization_6/moments/StopGradientStopGradientGmodel_1/transformer_block_3/layer_normalization_6/moments/mean:output:0*
T0*,
_output_shapes
:??????????'?
Kmodel_1/transformer_block_3/layer_normalization_6/moments/SquaredDifferenceSquaredDifference#model_1/transformer_block_3/add:z:0Omodel_1/transformer_block_3/layer_normalization_6/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????'?
Tmodel_1/transformer_block_3/layer_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
Bmodel_1/transformer_block_3/layer_normalization_6/moments/varianceMeanOmodel_1/transformer_block_3/layer_normalization_6/moments/SquaredDifference:z:0]model_1/transformer_block_3/layer_normalization_6/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????'*
	keep_dims(?
Amodel_1/transformer_block_3/layer_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?5?
?model_1/transformer_block_3/layer_normalization_6/batchnorm/addAddV2Kmodel_1/transformer_block_3/layer_normalization_6/moments/variance:output:0Jmodel_1/transformer_block_3/layer_normalization_6/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????'?
Amodel_1/transformer_block_3/layer_normalization_6/batchnorm/RsqrtRsqrtCmodel_1/transformer_block_3/layer_normalization_6/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????'?
Nmodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOpWmodel_1_transformer_block_3_layer_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
?model_1/transformer_block_3/layer_normalization_6/batchnorm/mulMulEmodel_1/transformer_block_3/layer_normalization_6/batchnorm/Rsqrt:y:0Vmodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
Amodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul_1Mul#model_1/transformer_block_3/add:z:0Cmodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????'?
Amodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul_2MulGmodel_1/transformer_block_3/layer_normalization_6/moments/mean:output:0Cmodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????'?
Jmodel_1/transformer_block_3/layer_normalization_6/batchnorm/ReadVariableOpReadVariableOpSmodel_1_transformer_block_3_layer_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
?model_1/transformer_block_3/layer_normalization_6/batchnorm/subSubRmodel_1/transformer_block_3/layer_normalization_6/batchnorm/ReadVariableOp:value:0Emodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????'?
Amodel_1/transformer_block_3/layer_normalization_6/batchnorm/add_1AddV2Emodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul_1:z:0Cmodel_1/transformer_block_3/layer_normalization_6/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????'?
Jmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/ReadVariableOpReadVariableOpSmodel_1_transformer_block_3_sequential_3_dense_24_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0?
@model_1/transformer_block_3/sequential_3/dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
@model_1/transformer_block_3/sequential_3/dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
Amodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/ShapeShapeEmodel_1/transformer_block_3/layer_normalization_6/batchnorm/add_1:z:0*
T0*
_output_shapes
:?
Imodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Dmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/GatherV2GatherV2Jmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/Shape:output:0Imodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/free:output:0Rmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Kmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Fmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/GatherV2_1GatherV2Jmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/Shape:output:0Imodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/axes:output:0Tmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Amodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
@model_1/transformer_block_3/sequential_3/dense_24/Tensordot/ProdProdMmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/GatherV2:output:0Jmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
Cmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Bmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/Prod_1ProdOmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/GatherV2_1:output:0Lmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
Gmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Bmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/concatConcatV2Imodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/free:output:0Imodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/axes:output:0Pmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Amodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/stackPackImodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/Prod:output:0Kmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
Emodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/transpose	TransposeEmodel_1/transformer_block_3/layer_normalization_6/batchnorm/add_1:z:0Kmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
Cmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/ReshapeReshapeImodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/transpose:y:0Jmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Bmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/MatMulMatMulLmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/Reshape:output:0Rmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Cmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Imodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Dmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/concat_1ConcatV2Mmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/GatherV2:output:0Lmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/Const_2:output:0Rmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
;model_1/transformer_block_3/sequential_3/dense_24/TensordotReshapeLmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/MatMul:product:0Mmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
Hmodel_1/transformer_block_3/sequential_3/dense_24/BiasAdd/ReadVariableOpReadVariableOpQmodel_1_transformer_block_3_sequential_3_dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
9model_1/transformer_block_3/sequential_3/dense_24/BiasAddBiasAddDmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot:output:0Pmodel_1/transformer_block_3/sequential_3/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
6model_1/transformer_block_3/sequential_3/dense_24/ReluReluBmodel_1/transformer_block_3/sequential_3/dense_24/BiasAdd:output:0*
T0*,
_output_shapes
:??????????'?
Jmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/ReadVariableOpReadVariableOpSmodel_1_transformer_block_3_sequential_3_dense_25_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0?
@model_1/transformer_block_3/sequential_3/dense_25/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
@model_1/transformer_block_3/sequential_3/dense_25/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
Amodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/ShapeShapeDmodel_1/transformer_block_3/sequential_3/dense_24/Relu:activations:0*
T0*
_output_shapes
:?
Imodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Dmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/GatherV2GatherV2Jmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/Shape:output:0Imodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/free:output:0Rmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Kmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Fmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/GatherV2_1GatherV2Jmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/Shape:output:0Imodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/axes:output:0Tmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Amodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
@model_1/transformer_block_3/sequential_3/dense_25/Tensordot/ProdProdMmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/GatherV2:output:0Jmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
Cmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Bmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/Prod_1ProdOmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/GatherV2_1:output:0Lmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
Gmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Bmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/concatConcatV2Imodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/free:output:0Imodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/axes:output:0Pmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Amodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/stackPackImodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/Prod:output:0Kmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
Emodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/transpose	TransposeDmodel_1/transformer_block_3/sequential_3/dense_24/Relu:activations:0Kmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
Cmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/ReshapeReshapeImodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/transpose:y:0Jmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Bmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/MatMulMatMulLmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/Reshape:output:0Rmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Cmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Imodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Dmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/concat_1ConcatV2Mmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/GatherV2:output:0Lmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/Const_2:output:0Rmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
;model_1/transformer_block_3/sequential_3/dense_25/TensordotReshapeLmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/MatMul:product:0Mmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
Hmodel_1/transformer_block_3/sequential_3/dense_25/BiasAdd/ReadVariableOpReadVariableOpQmodel_1_transformer_block_3_sequential_3_dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
9model_1/transformer_block_3/sequential_3/dense_25/BiasAddBiasAddDmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot:output:0Pmodel_1/transformer_block_3/sequential_3/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
.model_1/transformer_block_3/dropout_9/IdentityIdentityBmodel_1/transformer_block_3/sequential_3/dense_25/BiasAdd:output:0*
T0*,
_output_shapes
:??????????'?
!model_1/transformer_block_3/add_1AddV2Emodel_1/transformer_block_3/layer_normalization_6/batchnorm/add_1:z:07model_1/transformer_block_3/dropout_9/Identity:output:0*
T0*,
_output_shapes
:??????????'?
Pmodel_1/transformer_block_3/layer_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
>model_1/transformer_block_3/layer_normalization_7/moments/meanMean%model_1/transformer_block_3/add_1:z:0Ymodel_1/transformer_block_3/layer_normalization_7/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????'*
	keep_dims(?
Fmodel_1/transformer_block_3/layer_normalization_7/moments/StopGradientStopGradientGmodel_1/transformer_block_3/layer_normalization_7/moments/mean:output:0*
T0*,
_output_shapes
:??????????'?
Kmodel_1/transformer_block_3/layer_normalization_7/moments/SquaredDifferenceSquaredDifference%model_1/transformer_block_3/add_1:z:0Omodel_1/transformer_block_3/layer_normalization_7/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????'?
Tmodel_1/transformer_block_3/layer_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
Bmodel_1/transformer_block_3/layer_normalization_7/moments/varianceMeanOmodel_1/transformer_block_3/layer_normalization_7/moments/SquaredDifference:z:0]model_1/transformer_block_3/layer_normalization_7/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????'*
	keep_dims(?
Amodel_1/transformer_block_3/layer_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?5?
?model_1/transformer_block_3/layer_normalization_7/batchnorm/addAddV2Kmodel_1/transformer_block_3/layer_normalization_7/moments/variance:output:0Jmodel_1/transformer_block_3/layer_normalization_7/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????'?
Amodel_1/transformer_block_3/layer_normalization_7/batchnorm/RsqrtRsqrtCmodel_1/transformer_block_3/layer_normalization_7/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????'?
Nmodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOpWmodel_1_transformer_block_3_layer_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
?model_1/transformer_block_3/layer_normalization_7/batchnorm/mulMulEmodel_1/transformer_block_3/layer_normalization_7/batchnorm/Rsqrt:y:0Vmodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
Amodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul_1Mul%model_1/transformer_block_3/add_1:z:0Cmodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????'?
Amodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul_2MulGmodel_1/transformer_block_3/layer_normalization_7/moments/mean:output:0Cmodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????'?
Jmodel_1/transformer_block_3/layer_normalization_7/batchnorm/ReadVariableOpReadVariableOpSmodel_1_transformer_block_3_layer_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
?model_1/transformer_block_3/layer_normalization_7/batchnorm/subSubRmodel_1/transformer_block_3/layer_normalization_7/batchnorm/ReadVariableOp:value:0Emodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????'?
Amodel_1/transformer_block_3/layer_normalization_7/batchnorm/add_1AddV2Emodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul_1:z:0Cmodel_1/transformer_block_3/layer_normalization_7/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????'{
9model_1/global_average_pooling1d_3/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
'model_1/global_average_pooling1d_3/MeanMeanEmodel_1/transformer_block_3/layer_normalization_7/batchnorm/add_1:z:0Bmodel_1/global_average_pooling1d_3/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
model_1/dropout_10/IdentityIdentity0model_1/global_average_pooling1d_3/Mean:output:0*
T0*'
_output_shapes
:??????????
&model_1/dense_26/MatMul/ReadVariableOpReadVariableOp/model_1_dense_26_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
model_1/dense_26/MatMulMatMul$model_1/dropout_10/Identity:output:0.model_1/dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
'model_1/dense_26/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_1/dense_26/BiasAddBiasAdd!model_1/dense_26/MatMul:product:0/model_1/dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
model_1/dense_26/ReluRelu!model_1/dense_26/BiasAdd:output:0*
T0*'
_output_shapes
:?????????~
model_1/dropout_11/IdentityIdentity#model_1/dense_26/Relu:activations:0*
T0*'
_output_shapes
:??????????
&model_1/dense_27/MatMul/ReadVariableOpReadVariableOp/model_1_dense_27_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
model_1/dense_27/MatMulMatMul$model_1/dropout_11/Identity:output:0.model_1/dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
'model_1/dense_27/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_1/dense_27/BiasAddBiasAdd!model_1/dense_27/MatMul:product:0/model_1/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p
IdentityIdentity!model_1/dense_27/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp(^model_1/dense_26/BiasAdd/ReadVariableOp'^model_1/dense_26/MatMul/ReadVariableOp(^model_1/dense_27/BiasAdd/ReadVariableOp'^model_1/dense_27/MatMul/ReadVariableOpF^model_1/token_and_position_embedding_14/embedding_32/embedding_lookupF^model_1/token_and_position_embedding_14/embedding_33/embedding_lookupK^model_1/transformer_block_3/layer_normalization_6/batchnorm/ReadVariableOpO^model_1/transformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOpK^model_1/transformer_block_3/layer_normalization_7/batchnorm/ReadVariableOpO^model_1/transformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOpS^model_1/transformer_block_3/multi_head_attention_3/dense_20/BiasAdd/ReadVariableOpU^model_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/ReadVariableOpS^model_1/transformer_block_3/multi_head_attention_3/dense_21/BiasAdd/ReadVariableOpU^model_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/ReadVariableOpS^model_1/transformer_block_3/multi_head_attention_3/dense_22/BiasAdd/ReadVariableOpU^model_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/ReadVariableOpS^model_1/transformer_block_3/multi_head_attention_3/dense_23/BiasAdd/ReadVariableOpU^model_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/ReadVariableOpI^model_1/transformer_block_3/sequential_3/dense_24/BiasAdd/ReadVariableOpK^model_1/transformer_block_3/sequential_3/dense_24/Tensordot/ReadVariableOpI^model_1/transformer_block_3/sequential_3/dense_25/BiasAdd/ReadVariableOpK^model_1/transformer_block_3/sequential_3/dense_25/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:??????????': : : : : : : : : : : : : : : : : : : : : : 2R
'model_1/dense_26/BiasAdd/ReadVariableOp'model_1/dense_26/BiasAdd/ReadVariableOp2P
&model_1/dense_26/MatMul/ReadVariableOp&model_1/dense_26/MatMul/ReadVariableOp2R
'model_1/dense_27/BiasAdd/ReadVariableOp'model_1/dense_27/BiasAdd/ReadVariableOp2P
&model_1/dense_27/MatMul/ReadVariableOp&model_1/dense_27/MatMul/ReadVariableOp2?
Emodel_1/token_and_position_embedding_14/embedding_32/embedding_lookupEmodel_1/token_and_position_embedding_14/embedding_32/embedding_lookup2?
Emodel_1/token_and_position_embedding_14/embedding_33/embedding_lookupEmodel_1/token_and_position_embedding_14/embedding_33/embedding_lookup2?
Jmodel_1/transformer_block_3/layer_normalization_6/batchnorm/ReadVariableOpJmodel_1/transformer_block_3/layer_normalization_6/batchnorm/ReadVariableOp2?
Nmodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOpNmodel_1/transformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOp2?
Jmodel_1/transformer_block_3/layer_normalization_7/batchnorm/ReadVariableOpJmodel_1/transformer_block_3/layer_normalization_7/batchnorm/ReadVariableOp2?
Nmodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOpNmodel_1/transformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOp2?
Rmodel_1/transformer_block_3/multi_head_attention_3/dense_20/BiasAdd/ReadVariableOpRmodel_1/transformer_block_3/multi_head_attention_3/dense_20/BiasAdd/ReadVariableOp2?
Tmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/ReadVariableOpTmodel_1/transformer_block_3/multi_head_attention_3/dense_20/Tensordot/ReadVariableOp2?
Rmodel_1/transformer_block_3/multi_head_attention_3/dense_21/BiasAdd/ReadVariableOpRmodel_1/transformer_block_3/multi_head_attention_3/dense_21/BiasAdd/ReadVariableOp2?
Tmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/ReadVariableOpTmodel_1/transformer_block_3/multi_head_attention_3/dense_21/Tensordot/ReadVariableOp2?
Rmodel_1/transformer_block_3/multi_head_attention_3/dense_22/BiasAdd/ReadVariableOpRmodel_1/transformer_block_3/multi_head_attention_3/dense_22/BiasAdd/ReadVariableOp2?
Tmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/ReadVariableOpTmodel_1/transformer_block_3/multi_head_attention_3/dense_22/Tensordot/ReadVariableOp2?
Rmodel_1/transformer_block_3/multi_head_attention_3/dense_23/BiasAdd/ReadVariableOpRmodel_1/transformer_block_3/multi_head_attention_3/dense_23/BiasAdd/ReadVariableOp2?
Tmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/ReadVariableOpTmodel_1/transformer_block_3/multi_head_attention_3/dense_23/Tensordot/ReadVariableOp2?
Hmodel_1/transformer_block_3/sequential_3/dense_24/BiasAdd/ReadVariableOpHmodel_1/transformer_block_3/sequential_3/dense_24/BiasAdd/ReadVariableOp2?
Jmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/ReadVariableOpJmodel_1/transformer_block_3/sequential_3/dense_24/Tensordot/ReadVariableOp2?
Hmodel_1/transformer_block_3/sequential_3/dense_25/BiasAdd/ReadVariableOpHmodel_1/transformer_block_3/sequential_3/dense_25/BiasAdd/ReadVariableOp2?
Jmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/ReadVariableOpJmodel_1/transformer_block_3/sequential_3/dense_25/Tensordot/ReadVariableOp:V R
,
_output_shapes
:??????????'
"
_user_specified_name
input_17
?
F
*__inference_dropout_11_layer_call_fn_19567

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_17332`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_3_layer_call_fn_16862
dense_24_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_24_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????'*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_16851t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????'`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????': : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
,
_output_shapes
:??????????'
(
_user_specified_namedense_24_input
?
?
(__inference_dense_26_layer_call_fn_19551

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_26_layer_call_and_return_conditional_losses_17321o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_dense_24_layer_call_and_return_conditional_losses_16808

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????'f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:??????????'z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????': : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????'
 
_user_specified_nameinputs
?
?
G__inference_sequential_3_layer_call_and_return_conditional_losses_16963
dense_24_input 
dense_24_16952:
dense_24_16954: 
dense_25_16957:
dense_25_16959:
identity?? dense_24/StatefulPartitionedCall? dense_25/StatefulPartitionedCall?
 dense_24/StatefulPartitionedCallStatefulPartitionedCalldense_24_inputdense_24_16952dense_24_16954*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_16808?
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_16957dense_25_16959*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_25_layer_call_and_return_conditional_losses_16844}
IdentityIdentity)dense_25/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????'?
NoOpNoOp!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????': : : : 2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall:\ X
,
_output_shapes
:??????????'
(
_user_specified_namedense_24_input
?
?
C__inference_dense_25_layer_call_and_return_conditional_losses_19827

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'d
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:??????????'z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????': : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????'
 
_user_specified_nameinputs
??
?
N__inference_transformer_block_3_layer_call_and_return_conditional_losses_17767

inputsS
Amulti_head_attention_3_dense_20_tensordot_readvariableop_resource:M
?multi_head_attention_3_dense_20_biasadd_readvariableop_resource:S
Amulti_head_attention_3_dense_21_tensordot_readvariableop_resource:M
?multi_head_attention_3_dense_21_biasadd_readvariableop_resource:S
Amulti_head_attention_3_dense_22_tensordot_readvariableop_resource:M
?multi_head_attention_3_dense_22_biasadd_readvariableop_resource:S
Amulti_head_attention_3_dense_23_tensordot_readvariableop_resource:M
?multi_head_attention_3_dense_23_biasadd_readvariableop_resource:I
;layer_normalization_6_batchnorm_mul_readvariableop_resource:E
7layer_normalization_6_batchnorm_readvariableop_resource:I
7sequential_3_dense_24_tensordot_readvariableop_resource:C
5sequential_3_dense_24_biasadd_readvariableop_resource:I
7sequential_3_dense_25_tensordot_readvariableop_resource:C
5sequential_3_dense_25_biasadd_readvariableop_resource:I
;layer_normalization_7_batchnorm_mul_readvariableop_resource:E
7layer_normalization_7_batchnorm_readvariableop_resource:
identity??.layer_normalization_6/batchnorm/ReadVariableOp?2layer_normalization_6/batchnorm/mul/ReadVariableOp?.layer_normalization_7/batchnorm/ReadVariableOp?2layer_normalization_7/batchnorm/mul/ReadVariableOp?6multi_head_attention_3/dense_20/BiasAdd/ReadVariableOp?8multi_head_attention_3/dense_20/Tensordot/ReadVariableOp?6multi_head_attention_3/dense_21/BiasAdd/ReadVariableOp?8multi_head_attention_3/dense_21/Tensordot/ReadVariableOp?6multi_head_attention_3/dense_22/BiasAdd/ReadVariableOp?8multi_head_attention_3/dense_22/Tensordot/ReadVariableOp?6multi_head_attention_3/dense_23/BiasAdd/ReadVariableOp?8multi_head_attention_3/dense_23/Tensordot/ReadVariableOp?,sequential_3/dense_24/BiasAdd/ReadVariableOp?.sequential_3/dense_24/Tensordot/ReadVariableOp?,sequential_3/dense_25/BiasAdd/ReadVariableOp?.sequential_3/dense_25/Tensordot/ReadVariableOpR
multi_head_attention_3/ShapeShapeinputs*
T0*
_output_shapes
:t
*multi_head_attention_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,multi_head_attention_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,multi_head_attention_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$multi_head_attention_3/strided_sliceStridedSlice%multi_head_attention_3/Shape:output:03multi_head_attention_3/strided_slice/stack:output:05multi_head_attention_3/strided_slice/stack_1:output:05multi_head_attention_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
8multi_head_attention_3/dense_20/Tensordot/ReadVariableOpReadVariableOpAmulti_head_attention_3_dense_20_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0x
.multi_head_attention_3/dense_20/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
.multi_head_attention_3/dense_20/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       e
/multi_head_attention_3/dense_20/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:y
7multi_head_attention_3/dense_20/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_attention_3/dense_20/Tensordot/GatherV2GatherV28multi_head_attention_3/dense_20/Tensordot/Shape:output:07multi_head_attention_3/dense_20/Tensordot/free:output:0@multi_head_attention_3/dense_20/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9multi_head_attention_3/dense_20/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_attention_3/dense_20/Tensordot/GatherV2_1GatherV28multi_head_attention_3/dense_20/Tensordot/Shape:output:07multi_head_attention_3/dense_20/Tensordot/axes:output:0Bmulti_head_attention_3/dense_20/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/multi_head_attention_3/dense_20/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.multi_head_attention_3/dense_20/Tensordot/ProdProd;multi_head_attention_3/dense_20/Tensordot/GatherV2:output:08multi_head_attention_3/dense_20/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_attention_3/dense_20/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
0multi_head_attention_3/dense_20/Tensordot/Prod_1Prod=multi_head_attention_3/dense_20/Tensordot/GatherV2_1:output:0:multi_head_attention_3/dense_20/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5multi_head_attention_3/dense_20/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0multi_head_attention_3/dense_20/Tensordot/concatConcatV27multi_head_attention_3/dense_20/Tensordot/free:output:07multi_head_attention_3/dense_20/Tensordot/axes:output:0>multi_head_attention_3/dense_20/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
/multi_head_attention_3/dense_20/Tensordot/stackPack7multi_head_attention_3/dense_20/Tensordot/Prod:output:09multi_head_attention_3/dense_20/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
3multi_head_attention_3/dense_20/Tensordot/transpose	Transposeinputs9multi_head_attention_3/dense_20/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
1multi_head_attention_3/dense_20/Tensordot/ReshapeReshape7multi_head_attention_3/dense_20/Tensordot/transpose:y:08multi_head_attention_3/dense_20/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
0multi_head_attention_3/dense_20/Tensordot/MatMulMatMul:multi_head_attention_3/dense_20/Tensordot/Reshape:output:0@multi_head_attention_3/dense_20/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????{
1multi_head_attention_3/dense_20/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:y
7multi_head_attention_3/dense_20/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_attention_3/dense_20/Tensordot/concat_1ConcatV2;multi_head_attention_3/dense_20/Tensordot/GatherV2:output:0:multi_head_attention_3/dense_20/Tensordot/Const_2:output:0@multi_head_attention_3/dense_20/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
)multi_head_attention_3/dense_20/TensordotReshape:multi_head_attention_3/dense_20/Tensordot/MatMul:product:0;multi_head_attention_3/dense_20/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
6multi_head_attention_3/dense_20/BiasAdd/ReadVariableOpReadVariableOp?multi_head_attention_3_dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'multi_head_attention_3/dense_20/BiasAddBiasAdd2multi_head_attention_3/dense_20/Tensordot:output:0>multi_head_attention_3/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
8multi_head_attention_3/dense_21/Tensordot/ReadVariableOpReadVariableOpAmulti_head_attention_3_dense_21_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0x
.multi_head_attention_3/dense_21/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
.multi_head_attention_3/dense_21/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       e
/multi_head_attention_3/dense_21/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:y
7multi_head_attention_3/dense_21/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_attention_3/dense_21/Tensordot/GatherV2GatherV28multi_head_attention_3/dense_21/Tensordot/Shape:output:07multi_head_attention_3/dense_21/Tensordot/free:output:0@multi_head_attention_3/dense_21/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9multi_head_attention_3/dense_21/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_attention_3/dense_21/Tensordot/GatherV2_1GatherV28multi_head_attention_3/dense_21/Tensordot/Shape:output:07multi_head_attention_3/dense_21/Tensordot/axes:output:0Bmulti_head_attention_3/dense_21/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/multi_head_attention_3/dense_21/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.multi_head_attention_3/dense_21/Tensordot/ProdProd;multi_head_attention_3/dense_21/Tensordot/GatherV2:output:08multi_head_attention_3/dense_21/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_attention_3/dense_21/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
0multi_head_attention_3/dense_21/Tensordot/Prod_1Prod=multi_head_attention_3/dense_21/Tensordot/GatherV2_1:output:0:multi_head_attention_3/dense_21/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5multi_head_attention_3/dense_21/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0multi_head_attention_3/dense_21/Tensordot/concatConcatV27multi_head_attention_3/dense_21/Tensordot/free:output:07multi_head_attention_3/dense_21/Tensordot/axes:output:0>multi_head_attention_3/dense_21/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
/multi_head_attention_3/dense_21/Tensordot/stackPack7multi_head_attention_3/dense_21/Tensordot/Prod:output:09multi_head_attention_3/dense_21/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
3multi_head_attention_3/dense_21/Tensordot/transpose	Transposeinputs9multi_head_attention_3/dense_21/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
1multi_head_attention_3/dense_21/Tensordot/ReshapeReshape7multi_head_attention_3/dense_21/Tensordot/transpose:y:08multi_head_attention_3/dense_21/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
0multi_head_attention_3/dense_21/Tensordot/MatMulMatMul:multi_head_attention_3/dense_21/Tensordot/Reshape:output:0@multi_head_attention_3/dense_21/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????{
1multi_head_attention_3/dense_21/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:y
7multi_head_attention_3/dense_21/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_attention_3/dense_21/Tensordot/concat_1ConcatV2;multi_head_attention_3/dense_21/Tensordot/GatherV2:output:0:multi_head_attention_3/dense_21/Tensordot/Const_2:output:0@multi_head_attention_3/dense_21/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
)multi_head_attention_3/dense_21/TensordotReshape:multi_head_attention_3/dense_21/Tensordot/MatMul:product:0;multi_head_attention_3/dense_21/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
6multi_head_attention_3/dense_21/BiasAdd/ReadVariableOpReadVariableOp?multi_head_attention_3_dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'multi_head_attention_3/dense_21/BiasAddBiasAdd2multi_head_attention_3/dense_21/Tensordot:output:0>multi_head_attention_3/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
8multi_head_attention_3/dense_22/Tensordot/ReadVariableOpReadVariableOpAmulti_head_attention_3_dense_22_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0x
.multi_head_attention_3/dense_22/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
.multi_head_attention_3/dense_22/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       e
/multi_head_attention_3/dense_22/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:y
7multi_head_attention_3/dense_22/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_attention_3/dense_22/Tensordot/GatherV2GatherV28multi_head_attention_3/dense_22/Tensordot/Shape:output:07multi_head_attention_3/dense_22/Tensordot/free:output:0@multi_head_attention_3/dense_22/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9multi_head_attention_3/dense_22/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_attention_3/dense_22/Tensordot/GatherV2_1GatherV28multi_head_attention_3/dense_22/Tensordot/Shape:output:07multi_head_attention_3/dense_22/Tensordot/axes:output:0Bmulti_head_attention_3/dense_22/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/multi_head_attention_3/dense_22/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.multi_head_attention_3/dense_22/Tensordot/ProdProd;multi_head_attention_3/dense_22/Tensordot/GatherV2:output:08multi_head_attention_3/dense_22/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_attention_3/dense_22/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
0multi_head_attention_3/dense_22/Tensordot/Prod_1Prod=multi_head_attention_3/dense_22/Tensordot/GatherV2_1:output:0:multi_head_attention_3/dense_22/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5multi_head_attention_3/dense_22/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0multi_head_attention_3/dense_22/Tensordot/concatConcatV27multi_head_attention_3/dense_22/Tensordot/free:output:07multi_head_attention_3/dense_22/Tensordot/axes:output:0>multi_head_attention_3/dense_22/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
/multi_head_attention_3/dense_22/Tensordot/stackPack7multi_head_attention_3/dense_22/Tensordot/Prod:output:09multi_head_attention_3/dense_22/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
3multi_head_attention_3/dense_22/Tensordot/transpose	Transposeinputs9multi_head_attention_3/dense_22/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
1multi_head_attention_3/dense_22/Tensordot/ReshapeReshape7multi_head_attention_3/dense_22/Tensordot/transpose:y:08multi_head_attention_3/dense_22/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
0multi_head_attention_3/dense_22/Tensordot/MatMulMatMul:multi_head_attention_3/dense_22/Tensordot/Reshape:output:0@multi_head_attention_3/dense_22/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????{
1multi_head_attention_3/dense_22/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:y
7multi_head_attention_3/dense_22/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_attention_3/dense_22/Tensordot/concat_1ConcatV2;multi_head_attention_3/dense_22/Tensordot/GatherV2:output:0:multi_head_attention_3/dense_22/Tensordot/Const_2:output:0@multi_head_attention_3/dense_22/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
)multi_head_attention_3/dense_22/TensordotReshape:multi_head_attention_3/dense_22/Tensordot/MatMul:product:0;multi_head_attention_3/dense_22/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
6multi_head_attention_3/dense_22/BiasAdd/ReadVariableOpReadVariableOp?multi_head_attention_3_dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'multi_head_attention_3/dense_22/BiasAddBiasAdd2multi_head_attention_3/dense_22/Tensordot:output:0>multi_head_attention_3/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'q
&multi_head_attention_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????h
&multi_head_attention_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :h
&multi_head_attention_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
$multi_head_attention_3/Reshape/shapePack-multi_head_attention_3/strided_slice:output:0/multi_head_attention_3/Reshape/shape/1:output:0/multi_head_attention_3/Reshape/shape/2:output:0/multi_head_attention_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
multi_head_attention_3/ReshapeReshape0multi_head_attention_3/dense_20/BiasAdd:output:0-multi_head_attention_3/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????~
%multi_head_attention_3/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
 multi_head_attention_3/transpose	Transpose'multi_head_attention_3/Reshape:output:0.multi_head_attention_3/transpose/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????s
(multi_head_attention_3/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????j
(multi_head_attention_3/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :j
(multi_head_attention_3/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
&multi_head_attention_3/Reshape_1/shapePack-multi_head_attention_3/strided_slice:output:01multi_head_attention_3/Reshape_1/shape/1:output:01multi_head_attention_3/Reshape_1/shape/2:output:01multi_head_attention_3/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:?
 multi_head_attention_3/Reshape_1Reshape0multi_head_attention_3/dense_21/BiasAdd:output:0/multi_head_attention_3/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"???????????????????
'multi_head_attention_3/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
"multi_head_attention_3/transpose_1	Transpose)multi_head_attention_3/Reshape_1:output:00multi_head_attention_3/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????s
(multi_head_attention_3/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????j
(multi_head_attention_3/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :j
(multi_head_attention_3/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
&multi_head_attention_3/Reshape_2/shapePack-multi_head_attention_3/strided_slice:output:01multi_head_attention_3/Reshape_2/shape/1:output:01multi_head_attention_3/Reshape_2/shape/2:output:01multi_head_attention_3/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:?
 multi_head_attention_3/Reshape_2Reshape0multi_head_attention_3/dense_22/BiasAdd:output:0/multi_head_attention_3/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"???????????????????
'multi_head_attention_3/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
"multi_head_attention_3/transpose_2	Transpose)multi_head_attention_3/Reshape_2:output:00multi_head_attention_3/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"???????????????????
multi_head_attention_3/MatMulBatchMatMulV2$multi_head_attention_3/transpose:y:0&multi_head_attention_3/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(t
multi_head_attention_3/Shape_1Shape&multi_head_attention_3/transpose_1:y:0*
T0*
_output_shapes
:
,multi_head_attention_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????x
.multi_head_attention_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.multi_head_attention_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&multi_head_attention_3/strided_slice_1StridedSlice'multi_head_attention_3/Shape_1:output:05multi_head_attention_3/strided_slice_1/stack:output:07multi_head_attention_3/strided_slice_1/stack_1:output:07multi_head_attention_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
multi_head_attention_3/CastCast/multi_head_attention_3/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: e
multi_head_attention_3/SqrtSqrtmulti_head_attention_3/Cast:y:0*
T0*
_output_shapes
: ?
multi_head_attention_3/truedivRealDiv&multi_head_attention_3/MatMul:output:0multi_head_attention_3/Sqrt:y:0*
T0*A
_output_shapes/
-:+????????????????????????????
multi_head_attention_3/SoftmaxSoftmax"multi_head_attention_3/truediv:z:0*
T0*A
_output_shapes/
-:+????????????????????????????
multi_head_attention_3/MatMul_1BatchMatMulV2(multi_head_attention_3/Softmax:softmax:0&multi_head_attention_3/transpose_2:y:0*
T0*8
_output_shapes&
$:"???????????????????
'multi_head_attention_3/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
"multi_head_attention_3/transpose_3	Transpose(multi_head_attention_3/MatMul_1:output:00multi_head_attention_3/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????s
(multi_head_attention_3/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????j
(multi_head_attention_3/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
&multi_head_attention_3/Reshape_3/shapePack-multi_head_attention_3/strided_slice:output:01multi_head_attention_3/Reshape_3/shape/1:output:01multi_head_attention_3/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:?
 multi_head_attention_3/Reshape_3Reshape&multi_head_attention_3/transpose_3:y:0/multi_head_attention_3/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :???????????????????
8multi_head_attention_3/dense_23/Tensordot/ReadVariableOpReadVariableOpAmulti_head_attention_3_dense_23_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0x
.multi_head_attention_3/dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
.multi_head_attention_3/dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
/multi_head_attention_3/dense_23/Tensordot/ShapeShape)multi_head_attention_3/Reshape_3:output:0*
T0*
_output_shapes
:y
7multi_head_attention_3/dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_attention_3/dense_23/Tensordot/GatherV2GatherV28multi_head_attention_3/dense_23/Tensordot/Shape:output:07multi_head_attention_3/dense_23/Tensordot/free:output:0@multi_head_attention_3/dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9multi_head_attention_3/dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_attention_3/dense_23/Tensordot/GatherV2_1GatherV28multi_head_attention_3/dense_23/Tensordot/Shape:output:07multi_head_attention_3/dense_23/Tensordot/axes:output:0Bmulti_head_attention_3/dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/multi_head_attention_3/dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.multi_head_attention_3/dense_23/Tensordot/ProdProd;multi_head_attention_3/dense_23/Tensordot/GatherV2:output:08multi_head_attention_3/dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_attention_3/dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
0multi_head_attention_3/dense_23/Tensordot/Prod_1Prod=multi_head_attention_3/dense_23/Tensordot/GatherV2_1:output:0:multi_head_attention_3/dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5multi_head_attention_3/dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0multi_head_attention_3/dense_23/Tensordot/concatConcatV27multi_head_attention_3/dense_23/Tensordot/free:output:07multi_head_attention_3/dense_23/Tensordot/axes:output:0>multi_head_attention_3/dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
/multi_head_attention_3/dense_23/Tensordot/stackPack7multi_head_attention_3/dense_23/Tensordot/Prod:output:09multi_head_attention_3/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
3multi_head_attention_3/dense_23/Tensordot/transpose	Transpose)multi_head_attention_3/Reshape_3:output:09multi_head_attention_3/dense_23/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :???????????????????
1multi_head_attention_3/dense_23/Tensordot/ReshapeReshape7multi_head_attention_3/dense_23/Tensordot/transpose:y:08multi_head_attention_3/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
0multi_head_attention_3/dense_23/Tensordot/MatMulMatMul:multi_head_attention_3/dense_23/Tensordot/Reshape:output:0@multi_head_attention_3/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????{
1multi_head_attention_3/dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:y
7multi_head_attention_3/dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_attention_3/dense_23/Tensordot/concat_1ConcatV2;multi_head_attention_3/dense_23/Tensordot/GatherV2:output:0:multi_head_attention_3/dense_23/Tensordot/Const_2:output:0@multi_head_attention_3/dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
)multi_head_attention_3/dense_23/TensordotReshape:multi_head_attention_3/dense_23/Tensordot/MatMul:product:0;multi_head_attention_3/dense_23/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :???????????????????
6multi_head_attention_3/dense_23/BiasAdd/ReadVariableOpReadVariableOp?multi_head_attention_3_dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'multi_head_attention_3/dense_23/BiasAddBiasAdd2multi_head_attention_3/dense_23/Tensordot:output:0>multi_head_attention_3/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????\
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_8/dropout/MulMul0multi_head_attention_3/dense_23/BiasAdd:output:0 dropout_8/dropout/Const:output:0*
T0*4
_output_shapes"
 :??????????????????w
dropout_8/dropout/ShapeShape0multi_head_attention_3/dense_23/BiasAdd:output:0*
T0*
_output_shapes
:?
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype0e
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :???????????????????
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :???????????????????
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*4
_output_shapes"
 :??????????????????h
addAddV2inputsdropout_8/dropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????'~
4layer_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
"layer_normalization_6/moments/meanMeanadd:z:0=layer_normalization_6/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????'*
	keep_dims(?
*layer_normalization_6/moments/StopGradientStopGradient+layer_normalization_6/moments/mean:output:0*
T0*,
_output_shapes
:??????????'?
/layer_normalization_6/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_6/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????'?
8layer_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_6/moments/varianceMean3layer_normalization_6/moments/SquaredDifference:z:0Alayer_normalization_6/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????'*
	keep_dims(j
%layer_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?5?
#layer_normalization_6/batchnorm/addAddV2/layer_normalization_6/moments/variance:output:0.layer_normalization_6/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????'?
%layer_normalization_6/batchnorm/RsqrtRsqrt'layer_normalization_6/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????'?
2layer_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
#layer_normalization_6/batchnorm/mulMul)layer_normalization_6/batchnorm/Rsqrt:y:0:layer_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
%layer_normalization_6/batchnorm/mul_1Muladd:z:0'layer_normalization_6/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????'?
%layer_normalization_6/batchnorm/mul_2Mul+layer_normalization_6/moments/mean:output:0'layer_normalization_6/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????'?
.layer_normalization_6/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
#layer_normalization_6/batchnorm/subSub6layer_normalization_6/batchnorm/ReadVariableOp:value:0)layer_normalization_6/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????'?
%layer_normalization_6/batchnorm/add_1AddV2)layer_normalization_6/batchnorm/mul_1:z:0'layer_normalization_6/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????'?
.sequential_3/dense_24/Tensordot/ReadVariableOpReadVariableOp7sequential_3_dense_24_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0n
$sequential_3/dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:u
$sequential_3/dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ~
%sequential_3/dense_24/Tensordot/ShapeShape)layer_normalization_6/batchnorm/add_1:z:0*
T0*
_output_shapes
:o
-sequential_3/dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(sequential_3/dense_24/Tensordot/GatherV2GatherV2.sequential_3/dense_24/Tensordot/Shape:output:0-sequential_3/dense_24/Tensordot/free:output:06sequential_3/dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:q
/sequential_3/dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
*sequential_3/dense_24/Tensordot/GatherV2_1GatherV2.sequential_3/dense_24/Tensordot/Shape:output:0-sequential_3/dense_24/Tensordot/axes:output:08sequential_3/dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
%sequential_3/dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
$sequential_3/dense_24/Tensordot/ProdProd1sequential_3/dense_24/Tensordot/GatherV2:output:0.sequential_3/dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'sequential_3/dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
&sequential_3/dense_24/Tensordot/Prod_1Prod3sequential_3/dense_24/Tensordot/GatherV2_1:output:00sequential_3/dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+sequential_3/dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
&sequential_3/dense_24/Tensordot/concatConcatV2-sequential_3/dense_24/Tensordot/free:output:0-sequential_3/dense_24/Tensordot/axes:output:04sequential_3/dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
%sequential_3/dense_24/Tensordot/stackPack-sequential_3/dense_24/Tensordot/Prod:output:0/sequential_3/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
)sequential_3/dense_24/Tensordot/transpose	Transpose)layer_normalization_6/batchnorm/add_1:z:0/sequential_3/dense_24/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
'sequential_3/dense_24/Tensordot/ReshapeReshape-sequential_3/dense_24/Tensordot/transpose:y:0.sequential_3/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
&sequential_3/dense_24/Tensordot/MatMulMatMul0sequential_3/dense_24/Tensordot/Reshape:output:06sequential_3/dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????q
'sequential_3/dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:o
-sequential_3/dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(sequential_3/dense_24/Tensordot/concat_1ConcatV21sequential_3/dense_24/Tensordot/GatherV2:output:00sequential_3/dense_24/Tensordot/Const_2:output:06sequential_3/dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
sequential_3/dense_24/TensordotReshape0sequential_3/dense_24/Tensordot/MatMul:product:01sequential_3/dense_24/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
,sequential_3/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_3/dense_24/BiasAddBiasAdd(sequential_3/dense_24/Tensordot:output:04sequential_3/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
sequential_3/dense_24/ReluRelu&sequential_3/dense_24/BiasAdd:output:0*
T0*,
_output_shapes
:??????????'?
.sequential_3/dense_25/Tensordot/ReadVariableOpReadVariableOp7sequential_3_dense_25_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0n
$sequential_3/dense_25/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:u
$sequential_3/dense_25/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
%sequential_3/dense_25/Tensordot/ShapeShape(sequential_3/dense_24/Relu:activations:0*
T0*
_output_shapes
:o
-sequential_3/dense_25/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(sequential_3/dense_25/Tensordot/GatherV2GatherV2.sequential_3/dense_25/Tensordot/Shape:output:0-sequential_3/dense_25/Tensordot/free:output:06sequential_3/dense_25/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:q
/sequential_3/dense_25/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
*sequential_3/dense_25/Tensordot/GatherV2_1GatherV2.sequential_3/dense_25/Tensordot/Shape:output:0-sequential_3/dense_25/Tensordot/axes:output:08sequential_3/dense_25/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
%sequential_3/dense_25/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
$sequential_3/dense_25/Tensordot/ProdProd1sequential_3/dense_25/Tensordot/GatherV2:output:0.sequential_3/dense_25/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'sequential_3/dense_25/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
&sequential_3/dense_25/Tensordot/Prod_1Prod3sequential_3/dense_25/Tensordot/GatherV2_1:output:00sequential_3/dense_25/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+sequential_3/dense_25/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
&sequential_3/dense_25/Tensordot/concatConcatV2-sequential_3/dense_25/Tensordot/free:output:0-sequential_3/dense_25/Tensordot/axes:output:04sequential_3/dense_25/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
%sequential_3/dense_25/Tensordot/stackPack-sequential_3/dense_25/Tensordot/Prod:output:0/sequential_3/dense_25/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
)sequential_3/dense_25/Tensordot/transpose	Transpose(sequential_3/dense_24/Relu:activations:0/sequential_3/dense_25/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
'sequential_3/dense_25/Tensordot/ReshapeReshape-sequential_3/dense_25/Tensordot/transpose:y:0.sequential_3/dense_25/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
&sequential_3/dense_25/Tensordot/MatMulMatMul0sequential_3/dense_25/Tensordot/Reshape:output:06sequential_3/dense_25/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????q
'sequential_3/dense_25/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:o
-sequential_3/dense_25/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(sequential_3/dense_25/Tensordot/concat_1ConcatV21sequential_3/dense_25/Tensordot/GatherV2:output:00sequential_3/dense_25/Tensordot/Const_2:output:06sequential_3/dense_25/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
sequential_3/dense_25/TensordotReshape0sequential_3/dense_25/Tensordot/MatMul:product:01sequential_3/dense_25/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
,sequential_3/dense_25/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_3/dense_25/BiasAddBiasAdd(sequential_3/dense_25/Tensordot:output:04sequential_3/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'\
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_9/dropout/MulMul&sequential_3/dense_25/BiasAdd:output:0 dropout_9/dropout/Const:output:0*
T0*,
_output_shapes
:??????????'m
dropout_9/dropout/ShapeShape&sequential_3/dense_25/BiasAdd:output:0*
T0*
_output_shapes
:?
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????'*
dtype0e
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????'?
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????'?
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????'?
add_1AddV2)layer_normalization_6/batchnorm/add_1:z:0dropout_9/dropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????'~
4layer_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
"layer_normalization_7/moments/meanMean	add_1:z:0=layer_normalization_7/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????'*
	keep_dims(?
*layer_normalization_7/moments/StopGradientStopGradient+layer_normalization_7/moments/mean:output:0*
T0*,
_output_shapes
:??????????'?
/layer_normalization_7/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_7/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????'?
8layer_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_7/moments/varianceMean3layer_normalization_7/moments/SquaredDifference:z:0Alayer_normalization_7/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????'*
	keep_dims(j
%layer_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?5?
#layer_normalization_7/batchnorm/addAddV2/layer_normalization_7/moments/variance:output:0.layer_normalization_7/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????'?
%layer_normalization_7/batchnorm/RsqrtRsqrt'layer_normalization_7/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????'?
2layer_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
#layer_normalization_7/batchnorm/mulMul)layer_normalization_7/batchnorm/Rsqrt:y:0:layer_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
%layer_normalization_7/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_7/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????'?
%layer_normalization_7/batchnorm/mul_2Mul+layer_normalization_7/moments/mean:output:0'layer_normalization_7/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????'?
.layer_normalization_7/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
#layer_normalization_7/batchnorm/subSub6layer_normalization_7/batchnorm/ReadVariableOp:value:0)layer_normalization_7/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????'?
%layer_normalization_7/batchnorm/add_1AddV2)layer_normalization_7/batchnorm/mul_1:z:0'layer_normalization_7/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????'}
IdentityIdentity)layer_normalization_7/batchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:??????????'?
NoOpNoOp/^layer_normalization_6/batchnorm/ReadVariableOp3^layer_normalization_6/batchnorm/mul/ReadVariableOp/^layer_normalization_7/batchnorm/ReadVariableOp3^layer_normalization_7/batchnorm/mul/ReadVariableOp7^multi_head_attention_3/dense_20/BiasAdd/ReadVariableOp9^multi_head_attention_3/dense_20/Tensordot/ReadVariableOp7^multi_head_attention_3/dense_21/BiasAdd/ReadVariableOp9^multi_head_attention_3/dense_21/Tensordot/ReadVariableOp7^multi_head_attention_3/dense_22/BiasAdd/ReadVariableOp9^multi_head_attention_3/dense_22/Tensordot/ReadVariableOp7^multi_head_attention_3/dense_23/BiasAdd/ReadVariableOp9^multi_head_attention_3/dense_23/Tensordot/ReadVariableOp-^sequential_3/dense_24/BiasAdd/ReadVariableOp/^sequential_3/dense_24/Tensordot/ReadVariableOp-^sequential_3/dense_25/BiasAdd/ReadVariableOp/^sequential_3/dense_25/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:??????????': : : : : : : : : : : : : : : : 2`
.layer_normalization_6/batchnorm/ReadVariableOp.layer_normalization_6/batchnorm/ReadVariableOp2h
2layer_normalization_6/batchnorm/mul/ReadVariableOp2layer_normalization_6/batchnorm/mul/ReadVariableOp2`
.layer_normalization_7/batchnorm/ReadVariableOp.layer_normalization_7/batchnorm/ReadVariableOp2h
2layer_normalization_7/batchnorm/mul/ReadVariableOp2layer_normalization_7/batchnorm/mul/ReadVariableOp2p
6multi_head_attention_3/dense_20/BiasAdd/ReadVariableOp6multi_head_attention_3/dense_20/BiasAdd/ReadVariableOp2t
8multi_head_attention_3/dense_20/Tensordot/ReadVariableOp8multi_head_attention_3/dense_20/Tensordot/ReadVariableOp2p
6multi_head_attention_3/dense_21/BiasAdd/ReadVariableOp6multi_head_attention_3/dense_21/BiasAdd/ReadVariableOp2t
8multi_head_attention_3/dense_21/Tensordot/ReadVariableOp8multi_head_attention_3/dense_21/Tensordot/ReadVariableOp2p
6multi_head_attention_3/dense_22/BiasAdd/ReadVariableOp6multi_head_attention_3/dense_22/BiasAdd/ReadVariableOp2t
8multi_head_attention_3/dense_22/Tensordot/ReadVariableOp8multi_head_attention_3/dense_22/Tensordot/ReadVariableOp2p
6multi_head_attention_3/dense_23/BiasAdd/ReadVariableOp6multi_head_attention_3/dense_23/BiasAdd/ReadVariableOp2t
8multi_head_attention_3/dense_23/Tensordot/ReadVariableOp8multi_head_attention_3/dense_23/Tensordot/ReadVariableOp2\
,sequential_3/dense_24/BiasAdd/ReadVariableOp,sequential_3/dense_24/BiasAdd/ReadVariableOp2`
.sequential_3/dense_24/Tensordot/ReadVariableOp.sequential_3/dense_24/Tensordot/ReadVariableOp2\
,sequential_3/dense_25/BiasAdd/ReadVariableOp,sequential_3/dense_25/BiasAdd/ReadVariableOp2`
.sequential_3/dense_25/Tensordot/ReadVariableOp.sequential_3/dense_25/Tensordot/ReadVariableOp:T P
,
_output_shapes
:??????????'
 
_user_specified_nameinputs
?	
d
E__inference_dropout_11_layer_call_and_return_conditional_losses_17428

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
q
U__inference_global_average_pooling1d_3_layer_call_and_return_conditional_losses_16973

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
'__inference_model_1_layer_call_fn_18279

inputs
unknown:	?N
	unknown_0:	?
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_17918o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:??????????': : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????'
 
_user_specified_nameinputs
?
c
E__inference_dropout_11_layer_call_and_return_conditional_losses_17332

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
Ġ
?
B__inference_model_1_layer_call_and_return_conditional_losses_18879

inputsV
Ctoken_and_position_embedding_14_embedding_33_embedding_lookup_18581:	?NV
Ctoken_and_position_embedding_14_embedding_32_embedding_lookup_18587:	?g
Utransformer_block_3_multi_head_attention_3_dense_20_tensordot_readvariableop_resource:a
Stransformer_block_3_multi_head_attention_3_dense_20_biasadd_readvariableop_resource:g
Utransformer_block_3_multi_head_attention_3_dense_21_tensordot_readvariableop_resource:a
Stransformer_block_3_multi_head_attention_3_dense_21_biasadd_readvariableop_resource:g
Utransformer_block_3_multi_head_attention_3_dense_22_tensordot_readvariableop_resource:a
Stransformer_block_3_multi_head_attention_3_dense_22_biasadd_readvariableop_resource:g
Utransformer_block_3_multi_head_attention_3_dense_23_tensordot_readvariableop_resource:a
Stransformer_block_3_multi_head_attention_3_dense_23_biasadd_readvariableop_resource:]
Otransformer_block_3_layer_normalization_6_batchnorm_mul_readvariableop_resource:Y
Ktransformer_block_3_layer_normalization_6_batchnorm_readvariableop_resource:]
Ktransformer_block_3_sequential_3_dense_24_tensordot_readvariableop_resource:W
Itransformer_block_3_sequential_3_dense_24_biasadd_readvariableop_resource:]
Ktransformer_block_3_sequential_3_dense_25_tensordot_readvariableop_resource:W
Itransformer_block_3_sequential_3_dense_25_biasadd_readvariableop_resource:]
Otransformer_block_3_layer_normalization_7_batchnorm_mul_readvariableop_resource:Y
Ktransformer_block_3_layer_normalization_7_batchnorm_readvariableop_resource:9
'dense_26_matmul_readvariableop_resource:6
(dense_26_biasadd_readvariableop_resource:9
'dense_27_matmul_readvariableop_resource:6
(dense_27_biasadd_readvariableop_resource:
identity??dense_26/BiasAdd/ReadVariableOp?dense_26/MatMul/ReadVariableOp?dense_27/BiasAdd/ReadVariableOp?dense_27/MatMul/ReadVariableOp?=token_and_position_embedding_14/embedding_32/embedding_lookup?=token_and_position_embedding_14/embedding_33/embedding_lookup?Btransformer_block_3/layer_normalization_6/batchnorm/ReadVariableOp?Ftransformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOp?Btransformer_block_3/layer_normalization_7/batchnorm/ReadVariableOp?Ftransformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOp?Jtransformer_block_3/multi_head_attention_3/dense_20/BiasAdd/ReadVariableOp?Ltransformer_block_3/multi_head_attention_3/dense_20/Tensordot/ReadVariableOp?Jtransformer_block_3/multi_head_attention_3/dense_21/BiasAdd/ReadVariableOp?Ltransformer_block_3/multi_head_attention_3/dense_21/Tensordot/ReadVariableOp?Jtransformer_block_3/multi_head_attention_3/dense_22/BiasAdd/ReadVariableOp?Ltransformer_block_3/multi_head_attention_3/dense_22/Tensordot/ReadVariableOp?Jtransformer_block_3/multi_head_attention_3/dense_23/BiasAdd/ReadVariableOp?Ltransformer_block_3/multi_head_attention_3/dense_23/Tensordot/ReadVariableOp?@transformer_block_3/sequential_3/dense_24/BiasAdd/ReadVariableOp?Btransformer_block_3/sequential_3/dense_24/Tensordot/ReadVariableOp?@transformer_block_3/sequential_3/dense_25/BiasAdd/ReadVariableOp?Btransformer_block_3/sequential_3/dense_25/Tensordot/ReadVariableOp?
3token_and_position_embedding_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            ?
5token_and_position_embedding_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           ?
5token_and_position_embedding_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
-token_and_position_embedding_14/strided_sliceStridedSliceinputs<token_and_position_embedding_14/strided_slice/stack:output:0>token_and_position_embedding_14/strided_slice/stack_1:output:0>token_and_position_embedding_14/strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:??????????'*

begin_mask*
end_mask?
5token_and_position_embedding_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"           ?
7token_and_position_embedding_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           ?
7token_and_position_embedding_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
/token_and_position_embedding_14/strided_slice_1StridedSliceinputs>token_and_position_embedding_14/strided_slice_1/stack:output:0@token_and_position_embedding_14/strided_slice_1/stack_1:output:0@token_and_position_embedding_14/strided_slice_1/stack_2:output:0*
Index0*
T0*,
_output_shapes
:??????????'*

begin_mask*
end_mask~
-token_and_position_embedding_14/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
'token_and_position_embedding_14/ReshapeReshape6token_and_position_embedding_14/strided_slice:output:06token_and_position_embedding_14/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????'?
/token_and_position_embedding_14/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
)token_and_position_embedding_14/Reshape_1Reshape8token_and_position_embedding_14/strided_slice_1:output:08token_and_position_embedding_14/Reshape_1/shape:output:0*
T0*(
_output_shapes
:??????????'?
1token_and_position_embedding_14/embedding_33/CastCast2token_and_position_embedding_14/Reshape_1:output:0*

DstT0*

SrcT0*(
_output_shapes
:??????????'?
=token_and_position_embedding_14/embedding_33/embedding_lookupResourceGatherCtoken_and_position_embedding_14_embedding_33_embedding_lookup_185815token_and_position_embedding_14/embedding_33/Cast:y:0*
Tindices0*V
_classL
JHloc:@token_and_position_embedding_14/embedding_33/embedding_lookup/18581*,
_output_shapes
:??????????'*
dtype0?
Ftoken_and_position_embedding_14/embedding_33/embedding_lookup/IdentityIdentityFtoken_and_position_embedding_14/embedding_33/embedding_lookup:output:0*
T0*V
_classL
JHloc:@token_and_position_embedding_14/embedding_33/embedding_lookup/18581*,
_output_shapes
:??????????'?
Htoken_and_position_embedding_14/embedding_33/embedding_lookup/Identity_1IdentityOtoken_and_position_embedding_14/embedding_33/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????'?
1token_and_position_embedding_14/embedding_32/CastCast0token_and_position_embedding_14/Reshape:output:0*

DstT0*

SrcT0*(
_output_shapes
:??????????'?
=token_and_position_embedding_14/embedding_32/embedding_lookupResourceGatherCtoken_and_position_embedding_14_embedding_32_embedding_lookup_185875token_and_position_embedding_14/embedding_32/Cast:y:0*
Tindices0*V
_classL
JHloc:@token_and_position_embedding_14/embedding_32/embedding_lookup/18587*,
_output_shapes
:??????????'*
dtype0?
Ftoken_and_position_embedding_14/embedding_32/embedding_lookup/IdentityIdentityFtoken_and_position_embedding_14/embedding_32/embedding_lookup:output:0*
T0*V
_classL
JHloc:@token_and_position_embedding_14/embedding_32/embedding_lookup/18587*,
_output_shapes
:??????????'?
Htoken_and_position_embedding_14/embedding_32/embedding_lookup/Identity_1IdentityOtoken_and_position_embedding_14/embedding_32/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????'?
#token_and_position_embedding_14/addAddV2Qtoken_and_position_embedding_14/embedding_32/embedding_lookup/Identity_1:output:0Qtoken_and_position_embedding_14/embedding_33/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????'?
0transformer_block_3/multi_head_attention_3/ShapeShape'token_and_position_embedding_14/add:z:0*
T0*
_output_shapes
:?
>transformer_block_3/multi_head_attention_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
@transformer_block_3/multi_head_attention_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
@transformer_block_3/multi_head_attention_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
8transformer_block_3/multi_head_attention_3/strided_sliceStridedSlice9transformer_block_3/multi_head_attention_3/Shape:output:0Gtransformer_block_3/multi_head_attention_3/strided_slice/stack:output:0Itransformer_block_3/multi_head_attention_3/strided_slice/stack_1:output:0Itransformer_block_3/multi_head_attention_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Ltransformer_block_3/multi_head_attention_3/dense_20/Tensordot/ReadVariableOpReadVariableOpUtransformer_block_3_multi_head_attention_3_dense_20_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0?
Btransformer_block_3/multi_head_attention_3/dense_20/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
Btransformer_block_3/multi_head_attention_3/dense_20/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
Ctransformer_block_3/multi_head_attention_3/dense_20/Tensordot/ShapeShape'token_and_position_embedding_14/add:z:0*
T0*
_output_shapes
:?
Ktransformer_block_3/multi_head_attention_3/dense_20/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ftransformer_block_3/multi_head_attention_3/dense_20/Tensordot/GatherV2GatherV2Ltransformer_block_3/multi_head_attention_3/dense_20/Tensordot/Shape:output:0Ktransformer_block_3/multi_head_attention_3/dense_20/Tensordot/free:output:0Ttransformer_block_3/multi_head_attention_3/dense_20/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Mtransformer_block_3/multi_head_attention_3/dense_20/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Htransformer_block_3/multi_head_attention_3/dense_20/Tensordot/GatherV2_1GatherV2Ltransformer_block_3/multi_head_attention_3/dense_20/Tensordot/Shape:output:0Ktransformer_block_3/multi_head_attention_3/dense_20/Tensordot/axes:output:0Vtransformer_block_3/multi_head_attention_3/dense_20/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Ctransformer_block_3/multi_head_attention_3/dense_20/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Btransformer_block_3/multi_head_attention_3/dense_20/Tensordot/ProdProdOtransformer_block_3/multi_head_attention_3/dense_20/Tensordot/GatherV2:output:0Ltransformer_block_3/multi_head_attention_3/dense_20/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
Etransformer_block_3/multi_head_attention_3/dense_20/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Dtransformer_block_3/multi_head_attention_3/dense_20/Tensordot/Prod_1ProdQtransformer_block_3/multi_head_attention_3/dense_20/Tensordot/GatherV2_1:output:0Ntransformer_block_3/multi_head_attention_3/dense_20/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
Itransformer_block_3/multi_head_attention_3/dense_20/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Dtransformer_block_3/multi_head_attention_3/dense_20/Tensordot/concatConcatV2Ktransformer_block_3/multi_head_attention_3/dense_20/Tensordot/free:output:0Ktransformer_block_3/multi_head_attention_3/dense_20/Tensordot/axes:output:0Rtransformer_block_3/multi_head_attention_3/dense_20/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Ctransformer_block_3/multi_head_attention_3/dense_20/Tensordot/stackPackKtransformer_block_3/multi_head_attention_3/dense_20/Tensordot/Prod:output:0Mtransformer_block_3/multi_head_attention_3/dense_20/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
Gtransformer_block_3/multi_head_attention_3/dense_20/Tensordot/transpose	Transpose'token_and_position_embedding_14/add:z:0Mtransformer_block_3/multi_head_attention_3/dense_20/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
Etransformer_block_3/multi_head_attention_3/dense_20/Tensordot/ReshapeReshapeKtransformer_block_3/multi_head_attention_3/dense_20/Tensordot/transpose:y:0Ltransformer_block_3/multi_head_attention_3/dense_20/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Dtransformer_block_3/multi_head_attention_3/dense_20/Tensordot/MatMulMatMulNtransformer_block_3/multi_head_attention_3/dense_20/Tensordot/Reshape:output:0Ttransformer_block_3/multi_head_attention_3/dense_20/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Etransformer_block_3/multi_head_attention_3/dense_20/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Ktransformer_block_3/multi_head_attention_3/dense_20/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ftransformer_block_3/multi_head_attention_3/dense_20/Tensordot/concat_1ConcatV2Otransformer_block_3/multi_head_attention_3/dense_20/Tensordot/GatherV2:output:0Ntransformer_block_3/multi_head_attention_3/dense_20/Tensordot/Const_2:output:0Ttransformer_block_3/multi_head_attention_3/dense_20/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
=transformer_block_3/multi_head_attention_3/dense_20/TensordotReshapeNtransformer_block_3/multi_head_attention_3/dense_20/Tensordot/MatMul:product:0Otransformer_block_3/multi_head_attention_3/dense_20/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
Jtransformer_block_3/multi_head_attention_3/dense_20/BiasAdd/ReadVariableOpReadVariableOpStransformer_block_3_multi_head_attention_3_dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
;transformer_block_3/multi_head_attention_3/dense_20/BiasAddBiasAddFtransformer_block_3/multi_head_attention_3/dense_20/Tensordot:output:0Rtransformer_block_3/multi_head_attention_3/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
Ltransformer_block_3/multi_head_attention_3/dense_21/Tensordot/ReadVariableOpReadVariableOpUtransformer_block_3_multi_head_attention_3_dense_21_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0?
Btransformer_block_3/multi_head_attention_3/dense_21/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
Btransformer_block_3/multi_head_attention_3/dense_21/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
Ctransformer_block_3/multi_head_attention_3/dense_21/Tensordot/ShapeShape'token_and_position_embedding_14/add:z:0*
T0*
_output_shapes
:?
Ktransformer_block_3/multi_head_attention_3/dense_21/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ftransformer_block_3/multi_head_attention_3/dense_21/Tensordot/GatherV2GatherV2Ltransformer_block_3/multi_head_attention_3/dense_21/Tensordot/Shape:output:0Ktransformer_block_3/multi_head_attention_3/dense_21/Tensordot/free:output:0Ttransformer_block_3/multi_head_attention_3/dense_21/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Mtransformer_block_3/multi_head_attention_3/dense_21/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Htransformer_block_3/multi_head_attention_3/dense_21/Tensordot/GatherV2_1GatherV2Ltransformer_block_3/multi_head_attention_3/dense_21/Tensordot/Shape:output:0Ktransformer_block_3/multi_head_attention_3/dense_21/Tensordot/axes:output:0Vtransformer_block_3/multi_head_attention_3/dense_21/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Ctransformer_block_3/multi_head_attention_3/dense_21/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Btransformer_block_3/multi_head_attention_3/dense_21/Tensordot/ProdProdOtransformer_block_3/multi_head_attention_3/dense_21/Tensordot/GatherV2:output:0Ltransformer_block_3/multi_head_attention_3/dense_21/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
Etransformer_block_3/multi_head_attention_3/dense_21/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Dtransformer_block_3/multi_head_attention_3/dense_21/Tensordot/Prod_1ProdQtransformer_block_3/multi_head_attention_3/dense_21/Tensordot/GatherV2_1:output:0Ntransformer_block_3/multi_head_attention_3/dense_21/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
Itransformer_block_3/multi_head_attention_3/dense_21/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Dtransformer_block_3/multi_head_attention_3/dense_21/Tensordot/concatConcatV2Ktransformer_block_3/multi_head_attention_3/dense_21/Tensordot/free:output:0Ktransformer_block_3/multi_head_attention_3/dense_21/Tensordot/axes:output:0Rtransformer_block_3/multi_head_attention_3/dense_21/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Ctransformer_block_3/multi_head_attention_3/dense_21/Tensordot/stackPackKtransformer_block_3/multi_head_attention_3/dense_21/Tensordot/Prod:output:0Mtransformer_block_3/multi_head_attention_3/dense_21/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
Gtransformer_block_3/multi_head_attention_3/dense_21/Tensordot/transpose	Transpose'token_and_position_embedding_14/add:z:0Mtransformer_block_3/multi_head_attention_3/dense_21/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
Etransformer_block_3/multi_head_attention_3/dense_21/Tensordot/ReshapeReshapeKtransformer_block_3/multi_head_attention_3/dense_21/Tensordot/transpose:y:0Ltransformer_block_3/multi_head_attention_3/dense_21/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Dtransformer_block_3/multi_head_attention_3/dense_21/Tensordot/MatMulMatMulNtransformer_block_3/multi_head_attention_3/dense_21/Tensordot/Reshape:output:0Ttransformer_block_3/multi_head_attention_3/dense_21/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Etransformer_block_3/multi_head_attention_3/dense_21/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Ktransformer_block_3/multi_head_attention_3/dense_21/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ftransformer_block_3/multi_head_attention_3/dense_21/Tensordot/concat_1ConcatV2Otransformer_block_3/multi_head_attention_3/dense_21/Tensordot/GatherV2:output:0Ntransformer_block_3/multi_head_attention_3/dense_21/Tensordot/Const_2:output:0Ttransformer_block_3/multi_head_attention_3/dense_21/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
=transformer_block_3/multi_head_attention_3/dense_21/TensordotReshapeNtransformer_block_3/multi_head_attention_3/dense_21/Tensordot/MatMul:product:0Otransformer_block_3/multi_head_attention_3/dense_21/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
Jtransformer_block_3/multi_head_attention_3/dense_21/BiasAdd/ReadVariableOpReadVariableOpStransformer_block_3_multi_head_attention_3_dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
;transformer_block_3/multi_head_attention_3/dense_21/BiasAddBiasAddFtransformer_block_3/multi_head_attention_3/dense_21/Tensordot:output:0Rtransformer_block_3/multi_head_attention_3/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
Ltransformer_block_3/multi_head_attention_3/dense_22/Tensordot/ReadVariableOpReadVariableOpUtransformer_block_3_multi_head_attention_3_dense_22_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0?
Btransformer_block_3/multi_head_attention_3/dense_22/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
Btransformer_block_3/multi_head_attention_3/dense_22/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
Ctransformer_block_3/multi_head_attention_3/dense_22/Tensordot/ShapeShape'token_and_position_embedding_14/add:z:0*
T0*
_output_shapes
:?
Ktransformer_block_3/multi_head_attention_3/dense_22/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ftransformer_block_3/multi_head_attention_3/dense_22/Tensordot/GatherV2GatherV2Ltransformer_block_3/multi_head_attention_3/dense_22/Tensordot/Shape:output:0Ktransformer_block_3/multi_head_attention_3/dense_22/Tensordot/free:output:0Ttransformer_block_3/multi_head_attention_3/dense_22/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Mtransformer_block_3/multi_head_attention_3/dense_22/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Htransformer_block_3/multi_head_attention_3/dense_22/Tensordot/GatherV2_1GatherV2Ltransformer_block_3/multi_head_attention_3/dense_22/Tensordot/Shape:output:0Ktransformer_block_3/multi_head_attention_3/dense_22/Tensordot/axes:output:0Vtransformer_block_3/multi_head_attention_3/dense_22/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Ctransformer_block_3/multi_head_attention_3/dense_22/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Btransformer_block_3/multi_head_attention_3/dense_22/Tensordot/ProdProdOtransformer_block_3/multi_head_attention_3/dense_22/Tensordot/GatherV2:output:0Ltransformer_block_3/multi_head_attention_3/dense_22/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
Etransformer_block_3/multi_head_attention_3/dense_22/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Dtransformer_block_3/multi_head_attention_3/dense_22/Tensordot/Prod_1ProdQtransformer_block_3/multi_head_attention_3/dense_22/Tensordot/GatherV2_1:output:0Ntransformer_block_3/multi_head_attention_3/dense_22/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
Itransformer_block_3/multi_head_attention_3/dense_22/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Dtransformer_block_3/multi_head_attention_3/dense_22/Tensordot/concatConcatV2Ktransformer_block_3/multi_head_attention_3/dense_22/Tensordot/free:output:0Ktransformer_block_3/multi_head_attention_3/dense_22/Tensordot/axes:output:0Rtransformer_block_3/multi_head_attention_3/dense_22/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Ctransformer_block_3/multi_head_attention_3/dense_22/Tensordot/stackPackKtransformer_block_3/multi_head_attention_3/dense_22/Tensordot/Prod:output:0Mtransformer_block_3/multi_head_attention_3/dense_22/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
Gtransformer_block_3/multi_head_attention_3/dense_22/Tensordot/transpose	Transpose'token_and_position_embedding_14/add:z:0Mtransformer_block_3/multi_head_attention_3/dense_22/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
Etransformer_block_3/multi_head_attention_3/dense_22/Tensordot/ReshapeReshapeKtransformer_block_3/multi_head_attention_3/dense_22/Tensordot/transpose:y:0Ltransformer_block_3/multi_head_attention_3/dense_22/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Dtransformer_block_3/multi_head_attention_3/dense_22/Tensordot/MatMulMatMulNtransformer_block_3/multi_head_attention_3/dense_22/Tensordot/Reshape:output:0Ttransformer_block_3/multi_head_attention_3/dense_22/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Etransformer_block_3/multi_head_attention_3/dense_22/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Ktransformer_block_3/multi_head_attention_3/dense_22/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ftransformer_block_3/multi_head_attention_3/dense_22/Tensordot/concat_1ConcatV2Otransformer_block_3/multi_head_attention_3/dense_22/Tensordot/GatherV2:output:0Ntransformer_block_3/multi_head_attention_3/dense_22/Tensordot/Const_2:output:0Ttransformer_block_3/multi_head_attention_3/dense_22/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
=transformer_block_3/multi_head_attention_3/dense_22/TensordotReshapeNtransformer_block_3/multi_head_attention_3/dense_22/Tensordot/MatMul:product:0Otransformer_block_3/multi_head_attention_3/dense_22/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
Jtransformer_block_3/multi_head_attention_3/dense_22/BiasAdd/ReadVariableOpReadVariableOpStransformer_block_3_multi_head_attention_3_dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
;transformer_block_3/multi_head_attention_3/dense_22/BiasAddBiasAddFtransformer_block_3/multi_head_attention_3/dense_22/Tensordot:output:0Rtransformer_block_3/multi_head_attention_3/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
:transformer_block_3/multi_head_attention_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????|
:transformer_block_3/multi_head_attention_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :|
:transformer_block_3/multi_head_attention_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
8transformer_block_3/multi_head_attention_3/Reshape/shapePackAtransformer_block_3/multi_head_attention_3/strided_slice:output:0Ctransformer_block_3/multi_head_attention_3/Reshape/shape/1:output:0Ctransformer_block_3/multi_head_attention_3/Reshape/shape/2:output:0Ctransformer_block_3/multi_head_attention_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
2transformer_block_3/multi_head_attention_3/ReshapeReshapeDtransformer_block_3/multi_head_attention_3/dense_20/BiasAdd:output:0Atransformer_block_3/multi_head_attention_3/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"???????????????????
9transformer_block_3/multi_head_attention_3/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
4transformer_block_3/multi_head_attention_3/transpose	Transpose;transformer_block_3/multi_head_attention_3/Reshape:output:0Btransformer_block_3/multi_head_attention_3/transpose/perm:output:0*
T0*8
_output_shapes&
$:"???????????????????
<transformer_block_3/multi_head_attention_3/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????~
<transformer_block_3/multi_head_attention_3/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :~
<transformer_block_3/multi_head_attention_3/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
:transformer_block_3/multi_head_attention_3/Reshape_1/shapePackAtransformer_block_3/multi_head_attention_3/strided_slice:output:0Etransformer_block_3/multi_head_attention_3/Reshape_1/shape/1:output:0Etransformer_block_3/multi_head_attention_3/Reshape_1/shape/2:output:0Etransformer_block_3/multi_head_attention_3/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:?
4transformer_block_3/multi_head_attention_3/Reshape_1ReshapeDtransformer_block_3/multi_head_attention_3/dense_21/BiasAdd:output:0Ctransformer_block_3/multi_head_attention_3/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"???????????????????
;transformer_block_3/multi_head_attention_3/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
6transformer_block_3/multi_head_attention_3/transpose_1	Transpose=transformer_block_3/multi_head_attention_3/Reshape_1:output:0Dtransformer_block_3/multi_head_attention_3/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"???????????????????
<transformer_block_3/multi_head_attention_3/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????~
<transformer_block_3/multi_head_attention_3/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :~
<transformer_block_3/multi_head_attention_3/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
:transformer_block_3/multi_head_attention_3/Reshape_2/shapePackAtransformer_block_3/multi_head_attention_3/strided_slice:output:0Etransformer_block_3/multi_head_attention_3/Reshape_2/shape/1:output:0Etransformer_block_3/multi_head_attention_3/Reshape_2/shape/2:output:0Etransformer_block_3/multi_head_attention_3/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:?
4transformer_block_3/multi_head_attention_3/Reshape_2ReshapeDtransformer_block_3/multi_head_attention_3/dense_22/BiasAdd:output:0Ctransformer_block_3/multi_head_attention_3/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"???????????????????
;transformer_block_3/multi_head_attention_3/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
6transformer_block_3/multi_head_attention_3/transpose_2	Transpose=transformer_block_3/multi_head_attention_3/Reshape_2:output:0Dtransformer_block_3/multi_head_attention_3/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"???????????????????
1transformer_block_3/multi_head_attention_3/MatMulBatchMatMulV28transformer_block_3/multi_head_attention_3/transpose:y:0:transformer_block_3/multi_head_attention_3/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(?
2transformer_block_3/multi_head_attention_3/Shape_1Shape:transformer_block_3/multi_head_attention_3/transpose_1:y:0*
T0*
_output_shapes
:?
@transformer_block_3/multi_head_attention_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Btransformer_block_3/multi_head_attention_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Btransformer_block_3/multi_head_attention_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
:transformer_block_3/multi_head_attention_3/strided_slice_1StridedSlice;transformer_block_3/multi_head_attention_3/Shape_1:output:0Itransformer_block_3/multi_head_attention_3/strided_slice_1/stack:output:0Ktransformer_block_3/multi_head_attention_3/strided_slice_1/stack_1:output:0Ktransformer_block_3/multi_head_attention_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
/transformer_block_3/multi_head_attention_3/CastCastCtransformer_block_3/multi_head_attention_3/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
/transformer_block_3/multi_head_attention_3/SqrtSqrt3transformer_block_3/multi_head_attention_3/Cast:y:0*
T0*
_output_shapes
: ?
2transformer_block_3/multi_head_attention_3/truedivRealDiv:transformer_block_3/multi_head_attention_3/MatMul:output:03transformer_block_3/multi_head_attention_3/Sqrt:y:0*
T0*A
_output_shapes/
-:+????????????????????????????
2transformer_block_3/multi_head_attention_3/SoftmaxSoftmax6transformer_block_3/multi_head_attention_3/truediv:z:0*
T0*A
_output_shapes/
-:+????????????????????????????
3transformer_block_3/multi_head_attention_3/MatMul_1BatchMatMulV2<transformer_block_3/multi_head_attention_3/Softmax:softmax:0:transformer_block_3/multi_head_attention_3/transpose_2:y:0*
T0*8
_output_shapes&
$:"???????????????????
;transformer_block_3/multi_head_attention_3/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
6transformer_block_3/multi_head_attention_3/transpose_3	Transpose<transformer_block_3/multi_head_attention_3/MatMul_1:output:0Dtransformer_block_3/multi_head_attention_3/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"???????????????????
<transformer_block_3/multi_head_attention_3/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????~
<transformer_block_3/multi_head_attention_3/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
:transformer_block_3/multi_head_attention_3/Reshape_3/shapePackAtransformer_block_3/multi_head_attention_3/strided_slice:output:0Etransformer_block_3/multi_head_attention_3/Reshape_3/shape/1:output:0Etransformer_block_3/multi_head_attention_3/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:?
4transformer_block_3/multi_head_attention_3/Reshape_3Reshape:transformer_block_3/multi_head_attention_3/transpose_3:y:0Ctransformer_block_3/multi_head_attention_3/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :???????????????????
Ltransformer_block_3/multi_head_attention_3/dense_23/Tensordot/ReadVariableOpReadVariableOpUtransformer_block_3_multi_head_attention_3_dense_23_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0?
Btransformer_block_3/multi_head_attention_3/dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
Btransformer_block_3/multi_head_attention_3/dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
Ctransformer_block_3/multi_head_attention_3/dense_23/Tensordot/ShapeShape=transformer_block_3/multi_head_attention_3/Reshape_3:output:0*
T0*
_output_shapes
:?
Ktransformer_block_3/multi_head_attention_3/dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ftransformer_block_3/multi_head_attention_3/dense_23/Tensordot/GatherV2GatherV2Ltransformer_block_3/multi_head_attention_3/dense_23/Tensordot/Shape:output:0Ktransformer_block_3/multi_head_attention_3/dense_23/Tensordot/free:output:0Ttransformer_block_3/multi_head_attention_3/dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Mtransformer_block_3/multi_head_attention_3/dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Htransformer_block_3/multi_head_attention_3/dense_23/Tensordot/GatherV2_1GatherV2Ltransformer_block_3/multi_head_attention_3/dense_23/Tensordot/Shape:output:0Ktransformer_block_3/multi_head_attention_3/dense_23/Tensordot/axes:output:0Vtransformer_block_3/multi_head_attention_3/dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Ctransformer_block_3/multi_head_attention_3/dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Btransformer_block_3/multi_head_attention_3/dense_23/Tensordot/ProdProdOtransformer_block_3/multi_head_attention_3/dense_23/Tensordot/GatherV2:output:0Ltransformer_block_3/multi_head_attention_3/dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
Etransformer_block_3/multi_head_attention_3/dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Dtransformer_block_3/multi_head_attention_3/dense_23/Tensordot/Prod_1ProdQtransformer_block_3/multi_head_attention_3/dense_23/Tensordot/GatherV2_1:output:0Ntransformer_block_3/multi_head_attention_3/dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
Itransformer_block_3/multi_head_attention_3/dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Dtransformer_block_3/multi_head_attention_3/dense_23/Tensordot/concatConcatV2Ktransformer_block_3/multi_head_attention_3/dense_23/Tensordot/free:output:0Ktransformer_block_3/multi_head_attention_3/dense_23/Tensordot/axes:output:0Rtransformer_block_3/multi_head_attention_3/dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Ctransformer_block_3/multi_head_attention_3/dense_23/Tensordot/stackPackKtransformer_block_3/multi_head_attention_3/dense_23/Tensordot/Prod:output:0Mtransformer_block_3/multi_head_attention_3/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
Gtransformer_block_3/multi_head_attention_3/dense_23/Tensordot/transpose	Transpose=transformer_block_3/multi_head_attention_3/Reshape_3:output:0Mtransformer_block_3/multi_head_attention_3/dense_23/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :???????????????????
Etransformer_block_3/multi_head_attention_3/dense_23/Tensordot/ReshapeReshapeKtransformer_block_3/multi_head_attention_3/dense_23/Tensordot/transpose:y:0Ltransformer_block_3/multi_head_attention_3/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Dtransformer_block_3/multi_head_attention_3/dense_23/Tensordot/MatMulMatMulNtransformer_block_3/multi_head_attention_3/dense_23/Tensordot/Reshape:output:0Ttransformer_block_3/multi_head_attention_3/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Etransformer_block_3/multi_head_attention_3/dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Ktransformer_block_3/multi_head_attention_3/dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ftransformer_block_3/multi_head_attention_3/dense_23/Tensordot/concat_1ConcatV2Otransformer_block_3/multi_head_attention_3/dense_23/Tensordot/GatherV2:output:0Ntransformer_block_3/multi_head_attention_3/dense_23/Tensordot/Const_2:output:0Ttransformer_block_3/multi_head_attention_3/dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
=transformer_block_3/multi_head_attention_3/dense_23/TensordotReshapeNtransformer_block_3/multi_head_attention_3/dense_23/Tensordot/MatMul:product:0Otransformer_block_3/multi_head_attention_3/dense_23/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :???????????????????
Jtransformer_block_3/multi_head_attention_3/dense_23/BiasAdd/ReadVariableOpReadVariableOpStransformer_block_3_multi_head_attention_3_dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
;transformer_block_3/multi_head_attention_3/dense_23/BiasAddBiasAddFtransformer_block_3/multi_head_attention_3/dense_23/Tensordot:output:0Rtransformer_block_3/multi_head_attention_3/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????p
+transformer_block_3/dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
)transformer_block_3/dropout_8/dropout/MulMulDtransformer_block_3/multi_head_attention_3/dense_23/BiasAdd:output:04transformer_block_3/dropout_8/dropout/Const:output:0*
T0*4
_output_shapes"
 :???????????????????
+transformer_block_3/dropout_8/dropout/ShapeShapeDtransformer_block_3/multi_head_attention_3/dense_23/BiasAdd:output:0*
T0*
_output_shapes
:?
Btransformer_block_3/dropout_8/dropout/random_uniform/RandomUniformRandomUniform4transformer_block_3/dropout_8/dropout/Shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype0y
4transformer_block_3/dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
2transformer_block_3/dropout_8/dropout/GreaterEqualGreaterEqualKtransformer_block_3/dropout_8/dropout/random_uniform/RandomUniform:output:0=transformer_block_3/dropout_8/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :???????????????????
*transformer_block_3/dropout_8/dropout/CastCast6transformer_block_3/dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :???????????????????
+transformer_block_3/dropout_8/dropout/Mul_1Mul-transformer_block_3/dropout_8/dropout/Mul:z:0.transformer_block_3/dropout_8/dropout/Cast:y:0*
T0*4
_output_shapes"
 :???????????????????
transformer_block_3/addAddV2'token_and_position_embedding_14/add:z:0/transformer_block_3/dropout_8/dropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????'?
Htransformer_block_3/layer_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
6transformer_block_3/layer_normalization_6/moments/meanMeantransformer_block_3/add:z:0Qtransformer_block_3/layer_normalization_6/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????'*
	keep_dims(?
>transformer_block_3/layer_normalization_6/moments/StopGradientStopGradient?transformer_block_3/layer_normalization_6/moments/mean:output:0*
T0*,
_output_shapes
:??????????'?
Ctransformer_block_3/layer_normalization_6/moments/SquaredDifferenceSquaredDifferencetransformer_block_3/add:z:0Gtransformer_block_3/layer_normalization_6/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????'?
Ltransformer_block_3/layer_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
:transformer_block_3/layer_normalization_6/moments/varianceMeanGtransformer_block_3/layer_normalization_6/moments/SquaredDifference:z:0Utransformer_block_3/layer_normalization_6/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????'*
	keep_dims(~
9transformer_block_3/layer_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?5?
7transformer_block_3/layer_normalization_6/batchnorm/addAddV2Ctransformer_block_3/layer_normalization_6/moments/variance:output:0Btransformer_block_3/layer_normalization_6/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????'?
9transformer_block_3/layer_normalization_6/batchnorm/RsqrtRsqrt;transformer_block_3/layer_normalization_6/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????'?
Ftransformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_3_layer_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
7transformer_block_3/layer_normalization_6/batchnorm/mulMul=transformer_block_3/layer_normalization_6/batchnorm/Rsqrt:y:0Ntransformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
9transformer_block_3/layer_normalization_6/batchnorm/mul_1Multransformer_block_3/add:z:0;transformer_block_3/layer_normalization_6/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????'?
9transformer_block_3/layer_normalization_6/batchnorm/mul_2Mul?transformer_block_3/layer_normalization_6/moments/mean:output:0;transformer_block_3/layer_normalization_6/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????'?
Btransformer_block_3/layer_normalization_6/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_3_layer_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
7transformer_block_3/layer_normalization_6/batchnorm/subSubJtransformer_block_3/layer_normalization_6/batchnorm/ReadVariableOp:value:0=transformer_block_3/layer_normalization_6/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????'?
9transformer_block_3/layer_normalization_6/batchnorm/add_1AddV2=transformer_block_3/layer_normalization_6/batchnorm/mul_1:z:0;transformer_block_3/layer_normalization_6/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????'?
Btransformer_block_3/sequential_3/dense_24/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_3_sequential_3_dense_24_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0?
8transformer_block_3/sequential_3/dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
8transformer_block_3/sequential_3/dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
9transformer_block_3/sequential_3/dense_24/Tensordot/ShapeShape=transformer_block_3/layer_normalization_6/batchnorm/add_1:z:0*
T0*
_output_shapes
:?
Atransformer_block_3/sequential_3/dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
<transformer_block_3/sequential_3/dense_24/Tensordot/GatherV2GatherV2Btransformer_block_3/sequential_3/dense_24/Tensordot/Shape:output:0Atransformer_block_3/sequential_3/dense_24/Tensordot/free:output:0Jtransformer_block_3/sequential_3/dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Ctransformer_block_3/sequential_3/dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
>transformer_block_3/sequential_3/dense_24/Tensordot/GatherV2_1GatherV2Btransformer_block_3/sequential_3/dense_24/Tensordot/Shape:output:0Atransformer_block_3/sequential_3/dense_24/Tensordot/axes:output:0Ltransformer_block_3/sequential_3/dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
9transformer_block_3/sequential_3/dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
8transformer_block_3/sequential_3/dense_24/Tensordot/ProdProdEtransformer_block_3/sequential_3/dense_24/Tensordot/GatherV2:output:0Btransformer_block_3/sequential_3/dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
;transformer_block_3/sequential_3/dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
:transformer_block_3/sequential_3/dense_24/Tensordot/Prod_1ProdGtransformer_block_3/sequential_3/dense_24/Tensordot/GatherV2_1:output:0Dtransformer_block_3/sequential_3/dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
?transformer_block_3/sequential_3/dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
:transformer_block_3/sequential_3/dense_24/Tensordot/concatConcatV2Atransformer_block_3/sequential_3/dense_24/Tensordot/free:output:0Atransformer_block_3/sequential_3/dense_24/Tensordot/axes:output:0Htransformer_block_3/sequential_3/dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
9transformer_block_3/sequential_3/dense_24/Tensordot/stackPackAtransformer_block_3/sequential_3/dense_24/Tensordot/Prod:output:0Ctransformer_block_3/sequential_3/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
=transformer_block_3/sequential_3/dense_24/Tensordot/transpose	Transpose=transformer_block_3/layer_normalization_6/batchnorm/add_1:z:0Ctransformer_block_3/sequential_3/dense_24/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
;transformer_block_3/sequential_3/dense_24/Tensordot/ReshapeReshapeAtransformer_block_3/sequential_3/dense_24/Tensordot/transpose:y:0Btransformer_block_3/sequential_3/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
:transformer_block_3/sequential_3/dense_24/Tensordot/MatMulMatMulDtransformer_block_3/sequential_3/dense_24/Tensordot/Reshape:output:0Jtransformer_block_3/sequential_3/dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
;transformer_block_3/sequential_3/dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Atransformer_block_3/sequential_3/dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
<transformer_block_3/sequential_3/dense_24/Tensordot/concat_1ConcatV2Etransformer_block_3/sequential_3/dense_24/Tensordot/GatherV2:output:0Dtransformer_block_3/sequential_3/dense_24/Tensordot/Const_2:output:0Jtransformer_block_3/sequential_3/dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
3transformer_block_3/sequential_3/dense_24/TensordotReshapeDtransformer_block_3/sequential_3/dense_24/Tensordot/MatMul:product:0Etransformer_block_3/sequential_3/dense_24/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
@transformer_block_3/sequential_3/dense_24/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_3_sequential_3_dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
1transformer_block_3/sequential_3/dense_24/BiasAddBiasAdd<transformer_block_3/sequential_3/dense_24/Tensordot:output:0Htransformer_block_3/sequential_3/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
.transformer_block_3/sequential_3/dense_24/ReluRelu:transformer_block_3/sequential_3/dense_24/BiasAdd:output:0*
T0*,
_output_shapes
:??????????'?
Btransformer_block_3/sequential_3/dense_25/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_3_sequential_3_dense_25_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0?
8transformer_block_3/sequential_3/dense_25/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
8transformer_block_3/sequential_3/dense_25/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
9transformer_block_3/sequential_3/dense_25/Tensordot/ShapeShape<transformer_block_3/sequential_3/dense_24/Relu:activations:0*
T0*
_output_shapes
:?
Atransformer_block_3/sequential_3/dense_25/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
<transformer_block_3/sequential_3/dense_25/Tensordot/GatherV2GatherV2Btransformer_block_3/sequential_3/dense_25/Tensordot/Shape:output:0Atransformer_block_3/sequential_3/dense_25/Tensordot/free:output:0Jtransformer_block_3/sequential_3/dense_25/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Ctransformer_block_3/sequential_3/dense_25/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
>transformer_block_3/sequential_3/dense_25/Tensordot/GatherV2_1GatherV2Btransformer_block_3/sequential_3/dense_25/Tensordot/Shape:output:0Atransformer_block_3/sequential_3/dense_25/Tensordot/axes:output:0Ltransformer_block_3/sequential_3/dense_25/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
9transformer_block_3/sequential_3/dense_25/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
8transformer_block_3/sequential_3/dense_25/Tensordot/ProdProdEtransformer_block_3/sequential_3/dense_25/Tensordot/GatherV2:output:0Btransformer_block_3/sequential_3/dense_25/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
;transformer_block_3/sequential_3/dense_25/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
:transformer_block_3/sequential_3/dense_25/Tensordot/Prod_1ProdGtransformer_block_3/sequential_3/dense_25/Tensordot/GatherV2_1:output:0Dtransformer_block_3/sequential_3/dense_25/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
?transformer_block_3/sequential_3/dense_25/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
:transformer_block_3/sequential_3/dense_25/Tensordot/concatConcatV2Atransformer_block_3/sequential_3/dense_25/Tensordot/free:output:0Atransformer_block_3/sequential_3/dense_25/Tensordot/axes:output:0Htransformer_block_3/sequential_3/dense_25/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
9transformer_block_3/sequential_3/dense_25/Tensordot/stackPackAtransformer_block_3/sequential_3/dense_25/Tensordot/Prod:output:0Ctransformer_block_3/sequential_3/dense_25/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
=transformer_block_3/sequential_3/dense_25/Tensordot/transpose	Transpose<transformer_block_3/sequential_3/dense_24/Relu:activations:0Ctransformer_block_3/sequential_3/dense_25/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
;transformer_block_3/sequential_3/dense_25/Tensordot/ReshapeReshapeAtransformer_block_3/sequential_3/dense_25/Tensordot/transpose:y:0Btransformer_block_3/sequential_3/dense_25/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
:transformer_block_3/sequential_3/dense_25/Tensordot/MatMulMatMulDtransformer_block_3/sequential_3/dense_25/Tensordot/Reshape:output:0Jtransformer_block_3/sequential_3/dense_25/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
;transformer_block_3/sequential_3/dense_25/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Atransformer_block_3/sequential_3/dense_25/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
<transformer_block_3/sequential_3/dense_25/Tensordot/concat_1ConcatV2Etransformer_block_3/sequential_3/dense_25/Tensordot/GatherV2:output:0Dtransformer_block_3/sequential_3/dense_25/Tensordot/Const_2:output:0Jtransformer_block_3/sequential_3/dense_25/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
3transformer_block_3/sequential_3/dense_25/TensordotReshapeDtransformer_block_3/sequential_3/dense_25/Tensordot/MatMul:product:0Etransformer_block_3/sequential_3/dense_25/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
@transformer_block_3/sequential_3/dense_25/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_3_sequential_3_dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
1transformer_block_3/sequential_3/dense_25/BiasAddBiasAdd<transformer_block_3/sequential_3/dense_25/Tensordot:output:0Htransformer_block_3/sequential_3/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'p
+transformer_block_3/dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
)transformer_block_3/dropout_9/dropout/MulMul:transformer_block_3/sequential_3/dense_25/BiasAdd:output:04transformer_block_3/dropout_9/dropout/Const:output:0*
T0*,
_output_shapes
:??????????'?
+transformer_block_3/dropout_9/dropout/ShapeShape:transformer_block_3/sequential_3/dense_25/BiasAdd:output:0*
T0*
_output_shapes
:?
Btransformer_block_3/dropout_9/dropout/random_uniform/RandomUniformRandomUniform4transformer_block_3/dropout_9/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????'*
dtype0y
4transformer_block_3/dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
2transformer_block_3/dropout_9/dropout/GreaterEqualGreaterEqualKtransformer_block_3/dropout_9/dropout/random_uniform/RandomUniform:output:0=transformer_block_3/dropout_9/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????'?
*transformer_block_3/dropout_9/dropout/CastCast6transformer_block_3/dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????'?
+transformer_block_3/dropout_9/dropout/Mul_1Mul-transformer_block_3/dropout_9/dropout/Mul:z:0.transformer_block_3/dropout_9/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????'?
transformer_block_3/add_1AddV2=transformer_block_3/layer_normalization_6/batchnorm/add_1:z:0/transformer_block_3/dropout_9/dropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????'?
Htransformer_block_3/layer_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
6transformer_block_3/layer_normalization_7/moments/meanMeantransformer_block_3/add_1:z:0Qtransformer_block_3/layer_normalization_7/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????'*
	keep_dims(?
>transformer_block_3/layer_normalization_7/moments/StopGradientStopGradient?transformer_block_3/layer_normalization_7/moments/mean:output:0*
T0*,
_output_shapes
:??????????'?
Ctransformer_block_3/layer_normalization_7/moments/SquaredDifferenceSquaredDifferencetransformer_block_3/add_1:z:0Gtransformer_block_3/layer_normalization_7/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????'?
Ltransformer_block_3/layer_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
:transformer_block_3/layer_normalization_7/moments/varianceMeanGtransformer_block_3/layer_normalization_7/moments/SquaredDifference:z:0Utransformer_block_3/layer_normalization_7/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????'*
	keep_dims(~
9transformer_block_3/layer_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?5?
7transformer_block_3/layer_normalization_7/batchnorm/addAddV2Ctransformer_block_3/layer_normalization_7/moments/variance:output:0Btransformer_block_3/layer_normalization_7/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????'?
9transformer_block_3/layer_normalization_7/batchnorm/RsqrtRsqrt;transformer_block_3/layer_normalization_7/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????'?
Ftransformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_3_layer_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
7transformer_block_3/layer_normalization_7/batchnorm/mulMul=transformer_block_3/layer_normalization_7/batchnorm/Rsqrt:y:0Ntransformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
9transformer_block_3/layer_normalization_7/batchnorm/mul_1Multransformer_block_3/add_1:z:0;transformer_block_3/layer_normalization_7/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????'?
9transformer_block_3/layer_normalization_7/batchnorm/mul_2Mul?transformer_block_3/layer_normalization_7/moments/mean:output:0;transformer_block_3/layer_normalization_7/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????'?
Btransformer_block_3/layer_normalization_7/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_3_layer_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
7transformer_block_3/layer_normalization_7/batchnorm/subSubJtransformer_block_3/layer_normalization_7/batchnorm/ReadVariableOp:value:0=transformer_block_3/layer_normalization_7/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????'?
9transformer_block_3/layer_normalization_7/batchnorm/add_1AddV2=transformer_block_3/layer_normalization_7/batchnorm/mul_1:z:0;transformer_block_3/layer_normalization_7/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????'s
1global_average_pooling1d_3/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
global_average_pooling1d_3/MeanMean=transformer_block_3/layer_normalization_7/batchnorm/add_1:z:0:global_average_pooling1d_3/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????]
dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_10/dropout/MulMul(global_average_pooling1d_3/Mean:output:0!dropout_10/dropout/Const:output:0*
T0*'
_output_shapes
:?????????p
dropout_10/dropout/ShapeShape(global_average_pooling1d_3/Mean:output:0*
T0*
_output_shapes
:?
/dropout_10/dropout/random_uniform/RandomUniformRandomUniform!dropout_10/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0f
!dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_10/dropout/GreaterEqualGreaterEqual8dropout_10/dropout/random_uniform/RandomUniform:output:0*dropout_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
dropout_10/dropout/CastCast#dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
dropout_10/dropout/Mul_1Muldropout_10/dropout/Mul:z:0dropout_10/dropout/Cast:y:0*
T0*'
_output_shapes
:??????????
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_26/MatMulMatMuldropout_10/dropout/Mul_1:z:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*'
_output_shapes
:?????????]
dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_11/dropout/MulMuldense_26/Relu:activations:0!dropout_11/dropout/Const:output:0*
T0*'
_output_shapes
:?????????c
dropout_11/dropout/ShapeShapedense_26/Relu:activations:0*
T0*
_output_shapes
:?
/dropout_11/dropout/random_uniform/RandomUniformRandomUniform!dropout_11/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0f
!dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_11/dropout/GreaterEqualGreaterEqual8dropout_11/dropout/random_uniform/RandomUniform:output:0*dropout_11/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
dropout_11/dropout/CastCast#dropout_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
dropout_11/dropout/Mul_1Muldropout_11/dropout/Mul:z:0dropout_11/dropout/Cast:y:0*
T0*'
_output_shapes
:??????????
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_27/MatMulMatMuldropout_11/dropout/Mul_1:z:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_27/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp>^token_and_position_embedding_14/embedding_32/embedding_lookup>^token_and_position_embedding_14/embedding_33/embedding_lookupC^transformer_block_3/layer_normalization_6/batchnorm/ReadVariableOpG^transformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOpC^transformer_block_3/layer_normalization_7/batchnorm/ReadVariableOpG^transformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOpK^transformer_block_3/multi_head_attention_3/dense_20/BiasAdd/ReadVariableOpM^transformer_block_3/multi_head_attention_3/dense_20/Tensordot/ReadVariableOpK^transformer_block_3/multi_head_attention_3/dense_21/BiasAdd/ReadVariableOpM^transformer_block_3/multi_head_attention_3/dense_21/Tensordot/ReadVariableOpK^transformer_block_3/multi_head_attention_3/dense_22/BiasAdd/ReadVariableOpM^transformer_block_3/multi_head_attention_3/dense_22/Tensordot/ReadVariableOpK^transformer_block_3/multi_head_attention_3/dense_23/BiasAdd/ReadVariableOpM^transformer_block_3/multi_head_attention_3/dense_23/Tensordot/ReadVariableOpA^transformer_block_3/sequential_3/dense_24/BiasAdd/ReadVariableOpC^transformer_block_3/sequential_3/dense_24/Tensordot/ReadVariableOpA^transformer_block_3/sequential_3/dense_25/BiasAdd/ReadVariableOpC^transformer_block_3/sequential_3/dense_25/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:??????????': : : : : : : : : : : : : : : : : : : : : : 2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2~
=token_and_position_embedding_14/embedding_32/embedding_lookup=token_and_position_embedding_14/embedding_32/embedding_lookup2~
=token_and_position_embedding_14/embedding_33/embedding_lookup=token_and_position_embedding_14/embedding_33/embedding_lookup2?
Btransformer_block_3/layer_normalization_6/batchnorm/ReadVariableOpBtransformer_block_3/layer_normalization_6/batchnorm/ReadVariableOp2?
Ftransformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOpFtransformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOp2?
Btransformer_block_3/layer_normalization_7/batchnorm/ReadVariableOpBtransformer_block_3/layer_normalization_7/batchnorm/ReadVariableOp2?
Ftransformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOpFtransformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOp2?
Jtransformer_block_3/multi_head_attention_3/dense_20/BiasAdd/ReadVariableOpJtransformer_block_3/multi_head_attention_3/dense_20/BiasAdd/ReadVariableOp2?
Ltransformer_block_3/multi_head_attention_3/dense_20/Tensordot/ReadVariableOpLtransformer_block_3/multi_head_attention_3/dense_20/Tensordot/ReadVariableOp2?
Jtransformer_block_3/multi_head_attention_3/dense_21/BiasAdd/ReadVariableOpJtransformer_block_3/multi_head_attention_3/dense_21/BiasAdd/ReadVariableOp2?
Ltransformer_block_3/multi_head_attention_3/dense_21/Tensordot/ReadVariableOpLtransformer_block_3/multi_head_attention_3/dense_21/Tensordot/ReadVariableOp2?
Jtransformer_block_3/multi_head_attention_3/dense_22/BiasAdd/ReadVariableOpJtransformer_block_3/multi_head_attention_3/dense_22/BiasAdd/ReadVariableOp2?
Ltransformer_block_3/multi_head_attention_3/dense_22/Tensordot/ReadVariableOpLtransformer_block_3/multi_head_attention_3/dense_22/Tensordot/ReadVariableOp2?
Jtransformer_block_3/multi_head_attention_3/dense_23/BiasAdd/ReadVariableOpJtransformer_block_3/multi_head_attention_3/dense_23/BiasAdd/ReadVariableOp2?
Ltransformer_block_3/multi_head_attention_3/dense_23/Tensordot/ReadVariableOpLtransformer_block_3/multi_head_attention_3/dense_23/Tensordot/ReadVariableOp2?
@transformer_block_3/sequential_3/dense_24/BiasAdd/ReadVariableOp@transformer_block_3/sequential_3/dense_24/BiasAdd/ReadVariableOp2?
Btransformer_block_3/sequential_3/dense_24/Tensordot/ReadVariableOpBtransformer_block_3/sequential_3/dense_24/Tensordot/ReadVariableOp2?
@transformer_block_3/sequential_3/dense_25/BiasAdd/ReadVariableOp@transformer_block_3/sequential_3/dense_25/BiasAdd/ReadVariableOp2?
Btransformer_block_3/sequential_3/dense_25/Tensordot/ReadVariableOpBtransformer_block_3/sequential_3/dense_25/Tensordot/ReadVariableOp:T P
,
_output_shapes
:??????????'
 
_user_specified_nameinputs
?	
d
E__inference_dropout_10_layer_call_and_return_conditional_losses_17461

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
q
U__inference_global_average_pooling1d_3_layer_call_and_return_conditional_losses_19515

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :g
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????U
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????':T P
,
_output_shapes
:??????????'
 
_user_specified_nameinputs
?
?
,__inference_sequential_3_layer_call_fn_19634

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????'*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_16911t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????'`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????': : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????'
 
_user_specified_nameinputs
?
?
'__inference_model_1_layer_call_fn_17398
input_17
unknown:	?N
	unknown_0:	?
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_17unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_17351o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:??????????': : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:??????????'
"
_user_specified_name
input_17
??
?
N__inference_transformer_block_3_layer_call_and_return_conditional_losses_19235

inputsS
Amulti_head_attention_3_dense_20_tensordot_readvariableop_resource:M
?multi_head_attention_3_dense_20_biasadd_readvariableop_resource:S
Amulti_head_attention_3_dense_21_tensordot_readvariableop_resource:M
?multi_head_attention_3_dense_21_biasadd_readvariableop_resource:S
Amulti_head_attention_3_dense_22_tensordot_readvariableop_resource:M
?multi_head_attention_3_dense_22_biasadd_readvariableop_resource:S
Amulti_head_attention_3_dense_23_tensordot_readvariableop_resource:M
?multi_head_attention_3_dense_23_biasadd_readvariableop_resource:I
;layer_normalization_6_batchnorm_mul_readvariableop_resource:E
7layer_normalization_6_batchnorm_readvariableop_resource:I
7sequential_3_dense_24_tensordot_readvariableop_resource:C
5sequential_3_dense_24_biasadd_readvariableop_resource:I
7sequential_3_dense_25_tensordot_readvariableop_resource:C
5sequential_3_dense_25_biasadd_readvariableop_resource:I
;layer_normalization_7_batchnorm_mul_readvariableop_resource:E
7layer_normalization_7_batchnorm_readvariableop_resource:
identity??.layer_normalization_6/batchnorm/ReadVariableOp?2layer_normalization_6/batchnorm/mul/ReadVariableOp?.layer_normalization_7/batchnorm/ReadVariableOp?2layer_normalization_7/batchnorm/mul/ReadVariableOp?6multi_head_attention_3/dense_20/BiasAdd/ReadVariableOp?8multi_head_attention_3/dense_20/Tensordot/ReadVariableOp?6multi_head_attention_3/dense_21/BiasAdd/ReadVariableOp?8multi_head_attention_3/dense_21/Tensordot/ReadVariableOp?6multi_head_attention_3/dense_22/BiasAdd/ReadVariableOp?8multi_head_attention_3/dense_22/Tensordot/ReadVariableOp?6multi_head_attention_3/dense_23/BiasAdd/ReadVariableOp?8multi_head_attention_3/dense_23/Tensordot/ReadVariableOp?,sequential_3/dense_24/BiasAdd/ReadVariableOp?.sequential_3/dense_24/Tensordot/ReadVariableOp?,sequential_3/dense_25/BiasAdd/ReadVariableOp?.sequential_3/dense_25/Tensordot/ReadVariableOpR
multi_head_attention_3/ShapeShapeinputs*
T0*
_output_shapes
:t
*multi_head_attention_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,multi_head_attention_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,multi_head_attention_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$multi_head_attention_3/strided_sliceStridedSlice%multi_head_attention_3/Shape:output:03multi_head_attention_3/strided_slice/stack:output:05multi_head_attention_3/strided_slice/stack_1:output:05multi_head_attention_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
8multi_head_attention_3/dense_20/Tensordot/ReadVariableOpReadVariableOpAmulti_head_attention_3_dense_20_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0x
.multi_head_attention_3/dense_20/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
.multi_head_attention_3/dense_20/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       e
/multi_head_attention_3/dense_20/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:y
7multi_head_attention_3/dense_20/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_attention_3/dense_20/Tensordot/GatherV2GatherV28multi_head_attention_3/dense_20/Tensordot/Shape:output:07multi_head_attention_3/dense_20/Tensordot/free:output:0@multi_head_attention_3/dense_20/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9multi_head_attention_3/dense_20/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_attention_3/dense_20/Tensordot/GatherV2_1GatherV28multi_head_attention_3/dense_20/Tensordot/Shape:output:07multi_head_attention_3/dense_20/Tensordot/axes:output:0Bmulti_head_attention_3/dense_20/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/multi_head_attention_3/dense_20/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.multi_head_attention_3/dense_20/Tensordot/ProdProd;multi_head_attention_3/dense_20/Tensordot/GatherV2:output:08multi_head_attention_3/dense_20/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_attention_3/dense_20/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
0multi_head_attention_3/dense_20/Tensordot/Prod_1Prod=multi_head_attention_3/dense_20/Tensordot/GatherV2_1:output:0:multi_head_attention_3/dense_20/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5multi_head_attention_3/dense_20/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0multi_head_attention_3/dense_20/Tensordot/concatConcatV27multi_head_attention_3/dense_20/Tensordot/free:output:07multi_head_attention_3/dense_20/Tensordot/axes:output:0>multi_head_attention_3/dense_20/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
/multi_head_attention_3/dense_20/Tensordot/stackPack7multi_head_attention_3/dense_20/Tensordot/Prod:output:09multi_head_attention_3/dense_20/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
3multi_head_attention_3/dense_20/Tensordot/transpose	Transposeinputs9multi_head_attention_3/dense_20/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
1multi_head_attention_3/dense_20/Tensordot/ReshapeReshape7multi_head_attention_3/dense_20/Tensordot/transpose:y:08multi_head_attention_3/dense_20/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
0multi_head_attention_3/dense_20/Tensordot/MatMulMatMul:multi_head_attention_3/dense_20/Tensordot/Reshape:output:0@multi_head_attention_3/dense_20/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????{
1multi_head_attention_3/dense_20/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:y
7multi_head_attention_3/dense_20/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_attention_3/dense_20/Tensordot/concat_1ConcatV2;multi_head_attention_3/dense_20/Tensordot/GatherV2:output:0:multi_head_attention_3/dense_20/Tensordot/Const_2:output:0@multi_head_attention_3/dense_20/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
)multi_head_attention_3/dense_20/TensordotReshape:multi_head_attention_3/dense_20/Tensordot/MatMul:product:0;multi_head_attention_3/dense_20/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
6multi_head_attention_3/dense_20/BiasAdd/ReadVariableOpReadVariableOp?multi_head_attention_3_dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'multi_head_attention_3/dense_20/BiasAddBiasAdd2multi_head_attention_3/dense_20/Tensordot:output:0>multi_head_attention_3/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
8multi_head_attention_3/dense_21/Tensordot/ReadVariableOpReadVariableOpAmulti_head_attention_3_dense_21_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0x
.multi_head_attention_3/dense_21/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
.multi_head_attention_3/dense_21/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       e
/multi_head_attention_3/dense_21/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:y
7multi_head_attention_3/dense_21/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_attention_3/dense_21/Tensordot/GatherV2GatherV28multi_head_attention_3/dense_21/Tensordot/Shape:output:07multi_head_attention_3/dense_21/Tensordot/free:output:0@multi_head_attention_3/dense_21/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9multi_head_attention_3/dense_21/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_attention_3/dense_21/Tensordot/GatherV2_1GatherV28multi_head_attention_3/dense_21/Tensordot/Shape:output:07multi_head_attention_3/dense_21/Tensordot/axes:output:0Bmulti_head_attention_3/dense_21/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/multi_head_attention_3/dense_21/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.multi_head_attention_3/dense_21/Tensordot/ProdProd;multi_head_attention_3/dense_21/Tensordot/GatherV2:output:08multi_head_attention_3/dense_21/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_attention_3/dense_21/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
0multi_head_attention_3/dense_21/Tensordot/Prod_1Prod=multi_head_attention_3/dense_21/Tensordot/GatherV2_1:output:0:multi_head_attention_3/dense_21/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5multi_head_attention_3/dense_21/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0multi_head_attention_3/dense_21/Tensordot/concatConcatV27multi_head_attention_3/dense_21/Tensordot/free:output:07multi_head_attention_3/dense_21/Tensordot/axes:output:0>multi_head_attention_3/dense_21/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
/multi_head_attention_3/dense_21/Tensordot/stackPack7multi_head_attention_3/dense_21/Tensordot/Prod:output:09multi_head_attention_3/dense_21/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
3multi_head_attention_3/dense_21/Tensordot/transpose	Transposeinputs9multi_head_attention_3/dense_21/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
1multi_head_attention_3/dense_21/Tensordot/ReshapeReshape7multi_head_attention_3/dense_21/Tensordot/transpose:y:08multi_head_attention_3/dense_21/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
0multi_head_attention_3/dense_21/Tensordot/MatMulMatMul:multi_head_attention_3/dense_21/Tensordot/Reshape:output:0@multi_head_attention_3/dense_21/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????{
1multi_head_attention_3/dense_21/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:y
7multi_head_attention_3/dense_21/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_attention_3/dense_21/Tensordot/concat_1ConcatV2;multi_head_attention_3/dense_21/Tensordot/GatherV2:output:0:multi_head_attention_3/dense_21/Tensordot/Const_2:output:0@multi_head_attention_3/dense_21/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
)multi_head_attention_3/dense_21/TensordotReshape:multi_head_attention_3/dense_21/Tensordot/MatMul:product:0;multi_head_attention_3/dense_21/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
6multi_head_attention_3/dense_21/BiasAdd/ReadVariableOpReadVariableOp?multi_head_attention_3_dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'multi_head_attention_3/dense_21/BiasAddBiasAdd2multi_head_attention_3/dense_21/Tensordot:output:0>multi_head_attention_3/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
8multi_head_attention_3/dense_22/Tensordot/ReadVariableOpReadVariableOpAmulti_head_attention_3_dense_22_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0x
.multi_head_attention_3/dense_22/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
.multi_head_attention_3/dense_22/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       e
/multi_head_attention_3/dense_22/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:y
7multi_head_attention_3/dense_22/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_attention_3/dense_22/Tensordot/GatherV2GatherV28multi_head_attention_3/dense_22/Tensordot/Shape:output:07multi_head_attention_3/dense_22/Tensordot/free:output:0@multi_head_attention_3/dense_22/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9multi_head_attention_3/dense_22/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_attention_3/dense_22/Tensordot/GatherV2_1GatherV28multi_head_attention_3/dense_22/Tensordot/Shape:output:07multi_head_attention_3/dense_22/Tensordot/axes:output:0Bmulti_head_attention_3/dense_22/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/multi_head_attention_3/dense_22/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.multi_head_attention_3/dense_22/Tensordot/ProdProd;multi_head_attention_3/dense_22/Tensordot/GatherV2:output:08multi_head_attention_3/dense_22/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_attention_3/dense_22/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
0multi_head_attention_3/dense_22/Tensordot/Prod_1Prod=multi_head_attention_3/dense_22/Tensordot/GatherV2_1:output:0:multi_head_attention_3/dense_22/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5multi_head_attention_3/dense_22/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0multi_head_attention_3/dense_22/Tensordot/concatConcatV27multi_head_attention_3/dense_22/Tensordot/free:output:07multi_head_attention_3/dense_22/Tensordot/axes:output:0>multi_head_attention_3/dense_22/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
/multi_head_attention_3/dense_22/Tensordot/stackPack7multi_head_attention_3/dense_22/Tensordot/Prod:output:09multi_head_attention_3/dense_22/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
3multi_head_attention_3/dense_22/Tensordot/transpose	Transposeinputs9multi_head_attention_3/dense_22/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
1multi_head_attention_3/dense_22/Tensordot/ReshapeReshape7multi_head_attention_3/dense_22/Tensordot/transpose:y:08multi_head_attention_3/dense_22/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
0multi_head_attention_3/dense_22/Tensordot/MatMulMatMul:multi_head_attention_3/dense_22/Tensordot/Reshape:output:0@multi_head_attention_3/dense_22/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????{
1multi_head_attention_3/dense_22/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:y
7multi_head_attention_3/dense_22/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_attention_3/dense_22/Tensordot/concat_1ConcatV2;multi_head_attention_3/dense_22/Tensordot/GatherV2:output:0:multi_head_attention_3/dense_22/Tensordot/Const_2:output:0@multi_head_attention_3/dense_22/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
)multi_head_attention_3/dense_22/TensordotReshape:multi_head_attention_3/dense_22/Tensordot/MatMul:product:0;multi_head_attention_3/dense_22/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
6multi_head_attention_3/dense_22/BiasAdd/ReadVariableOpReadVariableOp?multi_head_attention_3_dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'multi_head_attention_3/dense_22/BiasAddBiasAdd2multi_head_attention_3/dense_22/Tensordot:output:0>multi_head_attention_3/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'q
&multi_head_attention_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????h
&multi_head_attention_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :h
&multi_head_attention_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
$multi_head_attention_3/Reshape/shapePack-multi_head_attention_3/strided_slice:output:0/multi_head_attention_3/Reshape/shape/1:output:0/multi_head_attention_3/Reshape/shape/2:output:0/multi_head_attention_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
multi_head_attention_3/ReshapeReshape0multi_head_attention_3/dense_20/BiasAdd:output:0-multi_head_attention_3/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????~
%multi_head_attention_3/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
 multi_head_attention_3/transpose	Transpose'multi_head_attention_3/Reshape:output:0.multi_head_attention_3/transpose/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????s
(multi_head_attention_3/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????j
(multi_head_attention_3/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :j
(multi_head_attention_3/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
&multi_head_attention_3/Reshape_1/shapePack-multi_head_attention_3/strided_slice:output:01multi_head_attention_3/Reshape_1/shape/1:output:01multi_head_attention_3/Reshape_1/shape/2:output:01multi_head_attention_3/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:?
 multi_head_attention_3/Reshape_1Reshape0multi_head_attention_3/dense_21/BiasAdd:output:0/multi_head_attention_3/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"???????????????????
'multi_head_attention_3/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
"multi_head_attention_3/transpose_1	Transpose)multi_head_attention_3/Reshape_1:output:00multi_head_attention_3/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????s
(multi_head_attention_3/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????j
(multi_head_attention_3/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :j
(multi_head_attention_3/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
&multi_head_attention_3/Reshape_2/shapePack-multi_head_attention_3/strided_slice:output:01multi_head_attention_3/Reshape_2/shape/1:output:01multi_head_attention_3/Reshape_2/shape/2:output:01multi_head_attention_3/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:?
 multi_head_attention_3/Reshape_2Reshape0multi_head_attention_3/dense_22/BiasAdd:output:0/multi_head_attention_3/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"???????????????????
'multi_head_attention_3/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
"multi_head_attention_3/transpose_2	Transpose)multi_head_attention_3/Reshape_2:output:00multi_head_attention_3/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"???????????????????
multi_head_attention_3/MatMulBatchMatMulV2$multi_head_attention_3/transpose:y:0&multi_head_attention_3/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(t
multi_head_attention_3/Shape_1Shape&multi_head_attention_3/transpose_1:y:0*
T0*
_output_shapes
:
,multi_head_attention_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????x
.multi_head_attention_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.multi_head_attention_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&multi_head_attention_3/strided_slice_1StridedSlice'multi_head_attention_3/Shape_1:output:05multi_head_attention_3/strided_slice_1/stack:output:07multi_head_attention_3/strided_slice_1/stack_1:output:07multi_head_attention_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
multi_head_attention_3/CastCast/multi_head_attention_3/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: e
multi_head_attention_3/SqrtSqrtmulti_head_attention_3/Cast:y:0*
T0*
_output_shapes
: ?
multi_head_attention_3/truedivRealDiv&multi_head_attention_3/MatMul:output:0multi_head_attention_3/Sqrt:y:0*
T0*A
_output_shapes/
-:+????????????????????????????
multi_head_attention_3/SoftmaxSoftmax"multi_head_attention_3/truediv:z:0*
T0*A
_output_shapes/
-:+????????????????????????????
multi_head_attention_3/MatMul_1BatchMatMulV2(multi_head_attention_3/Softmax:softmax:0&multi_head_attention_3/transpose_2:y:0*
T0*8
_output_shapes&
$:"???????????????????
'multi_head_attention_3/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
"multi_head_attention_3/transpose_3	Transpose(multi_head_attention_3/MatMul_1:output:00multi_head_attention_3/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????s
(multi_head_attention_3/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????j
(multi_head_attention_3/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
&multi_head_attention_3/Reshape_3/shapePack-multi_head_attention_3/strided_slice:output:01multi_head_attention_3/Reshape_3/shape/1:output:01multi_head_attention_3/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:?
 multi_head_attention_3/Reshape_3Reshape&multi_head_attention_3/transpose_3:y:0/multi_head_attention_3/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :???????????????????
8multi_head_attention_3/dense_23/Tensordot/ReadVariableOpReadVariableOpAmulti_head_attention_3_dense_23_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0x
.multi_head_attention_3/dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
.multi_head_attention_3/dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
/multi_head_attention_3/dense_23/Tensordot/ShapeShape)multi_head_attention_3/Reshape_3:output:0*
T0*
_output_shapes
:y
7multi_head_attention_3/dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_attention_3/dense_23/Tensordot/GatherV2GatherV28multi_head_attention_3/dense_23/Tensordot/Shape:output:07multi_head_attention_3/dense_23/Tensordot/free:output:0@multi_head_attention_3/dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9multi_head_attention_3/dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_attention_3/dense_23/Tensordot/GatherV2_1GatherV28multi_head_attention_3/dense_23/Tensordot/Shape:output:07multi_head_attention_3/dense_23/Tensordot/axes:output:0Bmulti_head_attention_3/dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/multi_head_attention_3/dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.multi_head_attention_3/dense_23/Tensordot/ProdProd;multi_head_attention_3/dense_23/Tensordot/GatherV2:output:08multi_head_attention_3/dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_attention_3/dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
0multi_head_attention_3/dense_23/Tensordot/Prod_1Prod=multi_head_attention_3/dense_23/Tensordot/GatherV2_1:output:0:multi_head_attention_3/dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5multi_head_attention_3/dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0multi_head_attention_3/dense_23/Tensordot/concatConcatV27multi_head_attention_3/dense_23/Tensordot/free:output:07multi_head_attention_3/dense_23/Tensordot/axes:output:0>multi_head_attention_3/dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
/multi_head_attention_3/dense_23/Tensordot/stackPack7multi_head_attention_3/dense_23/Tensordot/Prod:output:09multi_head_attention_3/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
3multi_head_attention_3/dense_23/Tensordot/transpose	Transpose)multi_head_attention_3/Reshape_3:output:09multi_head_attention_3/dense_23/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :???????????????????
1multi_head_attention_3/dense_23/Tensordot/ReshapeReshape7multi_head_attention_3/dense_23/Tensordot/transpose:y:08multi_head_attention_3/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
0multi_head_attention_3/dense_23/Tensordot/MatMulMatMul:multi_head_attention_3/dense_23/Tensordot/Reshape:output:0@multi_head_attention_3/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????{
1multi_head_attention_3/dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:y
7multi_head_attention_3/dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_attention_3/dense_23/Tensordot/concat_1ConcatV2;multi_head_attention_3/dense_23/Tensordot/GatherV2:output:0:multi_head_attention_3/dense_23/Tensordot/Const_2:output:0@multi_head_attention_3/dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
)multi_head_attention_3/dense_23/TensordotReshape:multi_head_attention_3/dense_23/Tensordot/MatMul:product:0;multi_head_attention_3/dense_23/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :???????????????????
6multi_head_attention_3/dense_23/BiasAdd/ReadVariableOpReadVariableOp?multi_head_attention_3_dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'multi_head_attention_3/dense_23/BiasAddBiasAdd2multi_head_attention_3/dense_23/Tensordot:output:0>multi_head_attention_3/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :???????????????????
dropout_8/IdentityIdentity0multi_head_attention_3/dense_23/BiasAdd:output:0*
T0*4
_output_shapes"
 :??????????????????h
addAddV2inputsdropout_8/Identity:output:0*
T0*,
_output_shapes
:??????????'~
4layer_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
"layer_normalization_6/moments/meanMeanadd:z:0=layer_normalization_6/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????'*
	keep_dims(?
*layer_normalization_6/moments/StopGradientStopGradient+layer_normalization_6/moments/mean:output:0*
T0*,
_output_shapes
:??????????'?
/layer_normalization_6/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_6/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????'?
8layer_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_6/moments/varianceMean3layer_normalization_6/moments/SquaredDifference:z:0Alayer_normalization_6/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????'*
	keep_dims(j
%layer_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?5?
#layer_normalization_6/batchnorm/addAddV2/layer_normalization_6/moments/variance:output:0.layer_normalization_6/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????'?
%layer_normalization_6/batchnorm/RsqrtRsqrt'layer_normalization_6/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????'?
2layer_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
#layer_normalization_6/batchnorm/mulMul)layer_normalization_6/batchnorm/Rsqrt:y:0:layer_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
%layer_normalization_6/batchnorm/mul_1Muladd:z:0'layer_normalization_6/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????'?
%layer_normalization_6/batchnorm/mul_2Mul+layer_normalization_6/moments/mean:output:0'layer_normalization_6/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????'?
.layer_normalization_6/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
#layer_normalization_6/batchnorm/subSub6layer_normalization_6/batchnorm/ReadVariableOp:value:0)layer_normalization_6/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????'?
%layer_normalization_6/batchnorm/add_1AddV2)layer_normalization_6/batchnorm/mul_1:z:0'layer_normalization_6/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????'?
.sequential_3/dense_24/Tensordot/ReadVariableOpReadVariableOp7sequential_3_dense_24_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0n
$sequential_3/dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:u
$sequential_3/dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ~
%sequential_3/dense_24/Tensordot/ShapeShape)layer_normalization_6/batchnorm/add_1:z:0*
T0*
_output_shapes
:o
-sequential_3/dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(sequential_3/dense_24/Tensordot/GatherV2GatherV2.sequential_3/dense_24/Tensordot/Shape:output:0-sequential_3/dense_24/Tensordot/free:output:06sequential_3/dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:q
/sequential_3/dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
*sequential_3/dense_24/Tensordot/GatherV2_1GatherV2.sequential_3/dense_24/Tensordot/Shape:output:0-sequential_3/dense_24/Tensordot/axes:output:08sequential_3/dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
%sequential_3/dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
$sequential_3/dense_24/Tensordot/ProdProd1sequential_3/dense_24/Tensordot/GatherV2:output:0.sequential_3/dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'sequential_3/dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
&sequential_3/dense_24/Tensordot/Prod_1Prod3sequential_3/dense_24/Tensordot/GatherV2_1:output:00sequential_3/dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+sequential_3/dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
&sequential_3/dense_24/Tensordot/concatConcatV2-sequential_3/dense_24/Tensordot/free:output:0-sequential_3/dense_24/Tensordot/axes:output:04sequential_3/dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
%sequential_3/dense_24/Tensordot/stackPack-sequential_3/dense_24/Tensordot/Prod:output:0/sequential_3/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
)sequential_3/dense_24/Tensordot/transpose	Transpose)layer_normalization_6/batchnorm/add_1:z:0/sequential_3/dense_24/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
'sequential_3/dense_24/Tensordot/ReshapeReshape-sequential_3/dense_24/Tensordot/transpose:y:0.sequential_3/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
&sequential_3/dense_24/Tensordot/MatMulMatMul0sequential_3/dense_24/Tensordot/Reshape:output:06sequential_3/dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????q
'sequential_3/dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:o
-sequential_3/dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(sequential_3/dense_24/Tensordot/concat_1ConcatV21sequential_3/dense_24/Tensordot/GatherV2:output:00sequential_3/dense_24/Tensordot/Const_2:output:06sequential_3/dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
sequential_3/dense_24/TensordotReshape0sequential_3/dense_24/Tensordot/MatMul:product:01sequential_3/dense_24/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
,sequential_3/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_3/dense_24/BiasAddBiasAdd(sequential_3/dense_24/Tensordot:output:04sequential_3/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
sequential_3/dense_24/ReluRelu&sequential_3/dense_24/BiasAdd:output:0*
T0*,
_output_shapes
:??????????'?
.sequential_3/dense_25/Tensordot/ReadVariableOpReadVariableOp7sequential_3_dense_25_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0n
$sequential_3/dense_25/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:u
$sequential_3/dense_25/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
%sequential_3/dense_25/Tensordot/ShapeShape(sequential_3/dense_24/Relu:activations:0*
T0*
_output_shapes
:o
-sequential_3/dense_25/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(sequential_3/dense_25/Tensordot/GatherV2GatherV2.sequential_3/dense_25/Tensordot/Shape:output:0-sequential_3/dense_25/Tensordot/free:output:06sequential_3/dense_25/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:q
/sequential_3/dense_25/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
*sequential_3/dense_25/Tensordot/GatherV2_1GatherV2.sequential_3/dense_25/Tensordot/Shape:output:0-sequential_3/dense_25/Tensordot/axes:output:08sequential_3/dense_25/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
%sequential_3/dense_25/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
$sequential_3/dense_25/Tensordot/ProdProd1sequential_3/dense_25/Tensordot/GatherV2:output:0.sequential_3/dense_25/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'sequential_3/dense_25/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
&sequential_3/dense_25/Tensordot/Prod_1Prod3sequential_3/dense_25/Tensordot/GatherV2_1:output:00sequential_3/dense_25/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+sequential_3/dense_25/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
&sequential_3/dense_25/Tensordot/concatConcatV2-sequential_3/dense_25/Tensordot/free:output:0-sequential_3/dense_25/Tensordot/axes:output:04sequential_3/dense_25/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
%sequential_3/dense_25/Tensordot/stackPack-sequential_3/dense_25/Tensordot/Prod:output:0/sequential_3/dense_25/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
)sequential_3/dense_25/Tensordot/transpose	Transpose(sequential_3/dense_24/Relu:activations:0/sequential_3/dense_25/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
'sequential_3/dense_25/Tensordot/ReshapeReshape-sequential_3/dense_25/Tensordot/transpose:y:0.sequential_3/dense_25/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
&sequential_3/dense_25/Tensordot/MatMulMatMul0sequential_3/dense_25/Tensordot/Reshape:output:06sequential_3/dense_25/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????q
'sequential_3/dense_25/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:o
-sequential_3/dense_25/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(sequential_3/dense_25/Tensordot/concat_1ConcatV21sequential_3/dense_25/Tensordot/GatherV2:output:00sequential_3/dense_25/Tensordot/Const_2:output:06sequential_3/dense_25/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
sequential_3/dense_25/TensordotReshape0sequential_3/dense_25/Tensordot/MatMul:product:01sequential_3/dense_25/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
,sequential_3/dense_25/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_3/dense_25/BiasAddBiasAdd(sequential_3/dense_25/Tensordot:output:04sequential_3/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'}
dropout_9/IdentityIdentity&sequential_3/dense_25/BiasAdd:output:0*
T0*,
_output_shapes
:??????????'?
add_1AddV2)layer_normalization_6/batchnorm/add_1:z:0dropout_9/Identity:output:0*
T0*,
_output_shapes
:??????????'~
4layer_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
"layer_normalization_7/moments/meanMean	add_1:z:0=layer_normalization_7/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????'*
	keep_dims(?
*layer_normalization_7/moments/StopGradientStopGradient+layer_normalization_7/moments/mean:output:0*
T0*,
_output_shapes
:??????????'?
/layer_normalization_7/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_7/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????'?
8layer_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_7/moments/varianceMean3layer_normalization_7/moments/SquaredDifference:z:0Alayer_normalization_7/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????'*
	keep_dims(j
%layer_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?5?
#layer_normalization_7/batchnorm/addAddV2/layer_normalization_7/moments/variance:output:0.layer_normalization_7/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????'?
%layer_normalization_7/batchnorm/RsqrtRsqrt'layer_normalization_7/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????'?
2layer_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
#layer_normalization_7/batchnorm/mulMul)layer_normalization_7/batchnorm/Rsqrt:y:0:layer_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
%layer_normalization_7/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_7/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????'?
%layer_normalization_7/batchnorm/mul_2Mul+layer_normalization_7/moments/mean:output:0'layer_normalization_7/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????'?
.layer_normalization_7/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
#layer_normalization_7/batchnorm/subSub6layer_normalization_7/batchnorm/ReadVariableOp:value:0)layer_normalization_7/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????'?
%layer_normalization_7/batchnorm/add_1AddV2)layer_normalization_7/batchnorm/mul_1:z:0'layer_normalization_7/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????'}
IdentityIdentity)layer_normalization_7/batchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:??????????'?
NoOpNoOp/^layer_normalization_6/batchnorm/ReadVariableOp3^layer_normalization_6/batchnorm/mul/ReadVariableOp/^layer_normalization_7/batchnorm/ReadVariableOp3^layer_normalization_7/batchnorm/mul/ReadVariableOp7^multi_head_attention_3/dense_20/BiasAdd/ReadVariableOp9^multi_head_attention_3/dense_20/Tensordot/ReadVariableOp7^multi_head_attention_3/dense_21/BiasAdd/ReadVariableOp9^multi_head_attention_3/dense_21/Tensordot/ReadVariableOp7^multi_head_attention_3/dense_22/BiasAdd/ReadVariableOp9^multi_head_attention_3/dense_22/Tensordot/ReadVariableOp7^multi_head_attention_3/dense_23/BiasAdd/ReadVariableOp9^multi_head_attention_3/dense_23/Tensordot/ReadVariableOp-^sequential_3/dense_24/BiasAdd/ReadVariableOp/^sequential_3/dense_24/Tensordot/ReadVariableOp-^sequential_3/dense_25/BiasAdd/ReadVariableOp/^sequential_3/dense_25/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:??????????': : : : : : : : : : : : : : : : 2`
.layer_normalization_6/batchnorm/ReadVariableOp.layer_normalization_6/batchnorm/ReadVariableOp2h
2layer_normalization_6/batchnorm/mul/ReadVariableOp2layer_normalization_6/batchnorm/mul/ReadVariableOp2`
.layer_normalization_7/batchnorm/ReadVariableOp.layer_normalization_7/batchnorm/ReadVariableOp2h
2layer_normalization_7/batchnorm/mul/ReadVariableOp2layer_normalization_7/batchnorm/mul/ReadVariableOp2p
6multi_head_attention_3/dense_20/BiasAdd/ReadVariableOp6multi_head_attention_3/dense_20/BiasAdd/ReadVariableOp2t
8multi_head_attention_3/dense_20/Tensordot/ReadVariableOp8multi_head_attention_3/dense_20/Tensordot/ReadVariableOp2p
6multi_head_attention_3/dense_21/BiasAdd/ReadVariableOp6multi_head_attention_3/dense_21/BiasAdd/ReadVariableOp2t
8multi_head_attention_3/dense_21/Tensordot/ReadVariableOp8multi_head_attention_3/dense_21/Tensordot/ReadVariableOp2p
6multi_head_attention_3/dense_22/BiasAdd/ReadVariableOp6multi_head_attention_3/dense_22/BiasAdd/ReadVariableOp2t
8multi_head_attention_3/dense_22/Tensordot/ReadVariableOp8multi_head_attention_3/dense_22/Tensordot/ReadVariableOp2p
6multi_head_attention_3/dense_23/BiasAdd/ReadVariableOp6multi_head_attention_3/dense_23/BiasAdd/ReadVariableOp2t
8multi_head_attention_3/dense_23/Tensordot/ReadVariableOp8multi_head_attention_3/dense_23/Tensordot/ReadVariableOp2\
,sequential_3/dense_24/BiasAdd/ReadVariableOp,sequential_3/dense_24/BiasAdd/ReadVariableOp2`
.sequential_3/dense_24/Tensordot/ReadVariableOp.sequential_3/dense_24/Tensordot/ReadVariableOp2\
,sequential_3/dense_25/BiasAdd/ReadVariableOp,sequential_3/dense_25/BiasAdd/ReadVariableOp2`
.sequential_3/dense_25/Tensordot/ReadVariableOp.sequential_3/dense_25/Tensordot/ReadVariableOp:T P
,
_output_shapes
:??????????'
 
_user_specified_nameinputs
??
?
B__inference_model_1_layer_call_and_return_conditional_losses_18565

inputsV
Ctoken_and_position_embedding_14_embedding_33_embedding_lookup_18295:	?NV
Ctoken_and_position_embedding_14_embedding_32_embedding_lookup_18301:	?g
Utransformer_block_3_multi_head_attention_3_dense_20_tensordot_readvariableop_resource:a
Stransformer_block_3_multi_head_attention_3_dense_20_biasadd_readvariableop_resource:g
Utransformer_block_3_multi_head_attention_3_dense_21_tensordot_readvariableop_resource:a
Stransformer_block_3_multi_head_attention_3_dense_21_biasadd_readvariableop_resource:g
Utransformer_block_3_multi_head_attention_3_dense_22_tensordot_readvariableop_resource:a
Stransformer_block_3_multi_head_attention_3_dense_22_biasadd_readvariableop_resource:g
Utransformer_block_3_multi_head_attention_3_dense_23_tensordot_readvariableop_resource:a
Stransformer_block_3_multi_head_attention_3_dense_23_biasadd_readvariableop_resource:]
Otransformer_block_3_layer_normalization_6_batchnorm_mul_readvariableop_resource:Y
Ktransformer_block_3_layer_normalization_6_batchnorm_readvariableop_resource:]
Ktransformer_block_3_sequential_3_dense_24_tensordot_readvariableop_resource:W
Itransformer_block_3_sequential_3_dense_24_biasadd_readvariableop_resource:]
Ktransformer_block_3_sequential_3_dense_25_tensordot_readvariableop_resource:W
Itransformer_block_3_sequential_3_dense_25_biasadd_readvariableop_resource:]
Otransformer_block_3_layer_normalization_7_batchnorm_mul_readvariableop_resource:Y
Ktransformer_block_3_layer_normalization_7_batchnorm_readvariableop_resource:9
'dense_26_matmul_readvariableop_resource:6
(dense_26_biasadd_readvariableop_resource:9
'dense_27_matmul_readvariableop_resource:6
(dense_27_biasadd_readvariableop_resource:
identity??dense_26/BiasAdd/ReadVariableOp?dense_26/MatMul/ReadVariableOp?dense_27/BiasAdd/ReadVariableOp?dense_27/MatMul/ReadVariableOp?=token_and_position_embedding_14/embedding_32/embedding_lookup?=token_and_position_embedding_14/embedding_33/embedding_lookup?Btransformer_block_3/layer_normalization_6/batchnorm/ReadVariableOp?Ftransformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOp?Btransformer_block_3/layer_normalization_7/batchnorm/ReadVariableOp?Ftransformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOp?Jtransformer_block_3/multi_head_attention_3/dense_20/BiasAdd/ReadVariableOp?Ltransformer_block_3/multi_head_attention_3/dense_20/Tensordot/ReadVariableOp?Jtransformer_block_3/multi_head_attention_3/dense_21/BiasAdd/ReadVariableOp?Ltransformer_block_3/multi_head_attention_3/dense_21/Tensordot/ReadVariableOp?Jtransformer_block_3/multi_head_attention_3/dense_22/BiasAdd/ReadVariableOp?Ltransformer_block_3/multi_head_attention_3/dense_22/Tensordot/ReadVariableOp?Jtransformer_block_3/multi_head_attention_3/dense_23/BiasAdd/ReadVariableOp?Ltransformer_block_3/multi_head_attention_3/dense_23/Tensordot/ReadVariableOp?@transformer_block_3/sequential_3/dense_24/BiasAdd/ReadVariableOp?Btransformer_block_3/sequential_3/dense_24/Tensordot/ReadVariableOp?@transformer_block_3/sequential_3/dense_25/BiasAdd/ReadVariableOp?Btransformer_block_3/sequential_3/dense_25/Tensordot/ReadVariableOp?
3token_and_position_embedding_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            ?
5token_and_position_embedding_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           ?
5token_and_position_embedding_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
-token_and_position_embedding_14/strided_sliceStridedSliceinputs<token_and_position_embedding_14/strided_slice/stack:output:0>token_and_position_embedding_14/strided_slice/stack_1:output:0>token_and_position_embedding_14/strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:??????????'*

begin_mask*
end_mask?
5token_and_position_embedding_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"           ?
7token_and_position_embedding_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           ?
7token_and_position_embedding_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
/token_and_position_embedding_14/strided_slice_1StridedSliceinputs>token_and_position_embedding_14/strided_slice_1/stack:output:0@token_and_position_embedding_14/strided_slice_1/stack_1:output:0@token_and_position_embedding_14/strided_slice_1/stack_2:output:0*
Index0*
T0*,
_output_shapes
:??????????'*

begin_mask*
end_mask~
-token_and_position_embedding_14/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
'token_and_position_embedding_14/ReshapeReshape6token_and_position_embedding_14/strided_slice:output:06token_and_position_embedding_14/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????'?
/token_and_position_embedding_14/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
)token_and_position_embedding_14/Reshape_1Reshape8token_and_position_embedding_14/strided_slice_1:output:08token_and_position_embedding_14/Reshape_1/shape:output:0*
T0*(
_output_shapes
:??????????'?
1token_and_position_embedding_14/embedding_33/CastCast2token_and_position_embedding_14/Reshape_1:output:0*

DstT0*

SrcT0*(
_output_shapes
:??????????'?
=token_and_position_embedding_14/embedding_33/embedding_lookupResourceGatherCtoken_and_position_embedding_14_embedding_33_embedding_lookup_182955token_and_position_embedding_14/embedding_33/Cast:y:0*
Tindices0*V
_classL
JHloc:@token_and_position_embedding_14/embedding_33/embedding_lookup/18295*,
_output_shapes
:??????????'*
dtype0?
Ftoken_and_position_embedding_14/embedding_33/embedding_lookup/IdentityIdentityFtoken_and_position_embedding_14/embedding_33/embedding_lookup:output:0*
T0*V
_classL
JHloc:@token_and_position_embedding_14/embedding_33/embedding_lookup/18295*,
_output_shapes
:??????????'?
Htoken_and_position_embedding_14/embedding_33/embedding_lookup/Identity_1IdentityOtoken_and_position_embedding_14/embedding_33/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????'?
1token_and_position_embedding_14/embedding_32/CastCast0token_and_position_embedding_14/Reshape:output:0*

DstT0*

SrcT0*(
_output_shapes
:??????????'?
=token_and_position_embedding_14/embedding_32/embedding_lookupResourceGatherCtoken_and_position_embedding_14_embedding_32_embedding_lookup_183015token_and_position_embedding_14/embedding_32/Cast:y:0*
Tindices0*V
_classL
JHloc:@token_and_position_embedding_14/embedding_32/embedding_lookup/18301*,
_output_shapes
:??????????'*
dtype0?
Ftoken_and_position_embedding_14/embedding_32/embedding_lookup/IdentityIdentityFtoken_and_position_embedding_14/embedding_32/embedding_lookup:output:0*
T0*V
_classL
JHloc:@token_and_position_embedding_14/embedding_32/embedding_lookup/18301*,
_output_shapes
:??????????'?
Htoken_and_position_embedding_14/embedding_32/embedding_lookup/Identity_1IdentityOtoken_and_position_embedding_14/embedding_32/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????'?
#token_and_position_embedding_14/addAddV2Qtoken_and_position_embedding_14/embedding_32/embedding_lookup/Identity_1:output:0Qtoken_and_position_embedding_14/embedding_33/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????'?
0transformer_block_3/multi_head_attention_3/ShapeShape'token_and_position_embedding_14/add:z:0*
T0*
_output_shapes
:?
>transformer_block_3/multi_head_attention_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
@transformer_block_3/multi_head_attention_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
@transformer_block_3/multi_head_attention_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
8transformer_block_3/multi_head_attention_3/strided_sliceStridedSlice9transformer_block_3/multi_head_attention_3/Shape:output:0Gtransformer_block_3/multi_head_attention_3/strided_slice/stack:output:0Itransformer_block_3/multi_head_attention_3/strided_slice/stack_1:output:0Itransformer_block_3/multi_head_attention_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Ltransformer_block_3/multi_head_attention_3/dense_20/Tensordot/ReadVariableOpReadVariableOpUtransformer_block_3_multi_head_attention_3_dense_20_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0?
Btransformer_block_3/multi_head_attention_3/dense_20/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
Btransformer_block_3/multi_head_attention_3/dense_20/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
Ctransformer_block_3/multi_head_attention_3/dense_20/Tensordot/ShapeShape'token_and_position_embedding_14/add:z:0*
T0*
_output_shapes
:?
Ktransformer_block_3/multi_head_attention_3/dense_20/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ftransformer_block_3/multi_head_attention_3/dense_20/Tensordot/GatherV2GatherV2Ltransformer_block_3/multi_head_attention_3/dense_20/Tensordot/Shape:output:0Ktransformer_block_3/multi_head_attention_3/dense_20/Tensordot/free:output:0Ttransformer_block_3/multi_head_attention_3/dense_20/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Mtransformer_block_3/multi_head_attention_3/dense_20/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Htransformer_block_3/multi_head_attention_3/dense_20/Tensordot/GatherV2_1GatherV2Ltransformer_block_3/multi_head_attention_3/dense_20/Tensordot/Shape:output:0Ktransformer_block_3/multi_head_attention_3/dense_20/Tensordot/axes:output:0Vtransformer_block_3/multi_head_attention_3/dense_20/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Ctransformer_block_3/multi_head_attention_3/dense_20/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Btransformer_block_3/multi_head_attention_3/dense_20/Tensordot/ProdProdOtransformer_block_3/multi_head_attention_3/dense_20/Tensordot/GatherV2:output:0Ltransformer_block_3/multi_head_attention_3/dense_20/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
Etransformer_block_3/multi_head_attention_3/dense_20/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Dtransformer_block_3/multi_head_attention_3/dense_20/Tensordot/Prod_1ProdQtransformer_block_3/multi_head_attention_3/dense_20/Tensordot/GatherV2_1:output:0Ntransformer_block_3/multi_head_attention_3/dense_20/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
Itransformer_block_3/multi_head_attention_3/dense_20/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Dtransformer_block_3/multi_head_attention_3/dense_20/Tensordot/concatConcatV2Ktransformer_block_3/multi_head_attention_3/dense_20/Tensordot/free:output:0Ktransformer_block_3/multi_head_attention_3/dense_20/Tensordot/axes:output:0Rtransformer_block_3/multi_head_attention_3/dense_20/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Ctransformer_block_3/multi_head_attention_3/dense_20/Tensordot/stackPackKtransformer_block_3/multi_head_attention_3/dense_20/Tensordot/Prod:output:0Mtransformer_block_3/multi_head_attention_3/dense_20/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
Gtransformer_block_3/multi_head_attention_3/dense_20/Tensordot/transpose	Transpose'token_and_position_embedding_14/add:z:0Mtransformer_block_3/multi_head_attention_3/dense_20/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
Etransformer_block_3/multi_head_attention_3/dense_20/Tensordot/ReshapeReshapeKtransformer_block_3/multi_head_attention_3/dense_20/Tensordot/transpose:y:0Ltransformer_block_3/multi_head_attention_3/dense_20/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Dtransformer_block_3/multi_head_attention_3/dense_20/Tensordot/MatMulMatMulNtransformer_block_3/multi_head_attention_3/dense_20/Tensordot/Reshape:output:0Ttransformer_block_3/multi_head_attention_3/dense_20/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Etransformer_block_3/multi_head_attention_3/dense_20/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Ktransformer_block_3/multi_head_attention_3/dense_20/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ftransformer_block_3/multi_head_attention_3/dense_20/Tensordot/concat_1ConcatV2Otransformer_block_3/multi_head_attention_3/dense_20/Tensordot/GatherV2:output:0Ntransformer_block_3/multi_head_attention_3/dense_20/Tensordot/Const_2:output:0Ttransformer_block_3/multi_head_attention_3/dense_20/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
=transformer_block_3/multi_head_attention_3/dense_20/TensordotReshapeNtransformer_block_3/multi_head_attention_3/dense_20/Tensordot/MatMul:product:0Otransformer_block_3/multi_head_attention_3/dense_20/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
Jtransformer_block_3/multi_head_attention_3/dense_20/BiasAdd/ReadVariableOpReadVariableOpStransformer_block_3_multi_head_attention_3_dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
;transformer_block_3/multi_head_attention_3/dense_20/BiasAddBiasAddFtransformer_block_3/multi_head_attention_3/dense_20/Tensordot:output:0Rtransformer_block_3/multi_head_attention_3/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
Ltransformer_block_3/multi_head_attention_3/dense_21/Tensordot/ReadVariableOpReadVariableOpUtransformer_block_3_multi_head_attention_3_dense_21_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0?
Btransformer_block_3/multi_head_attention_3/dense_21/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
Btransformer_block_3/multi_head_attention_3/dense_21/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
Ctransformer_block_3/multi_head_attention_3/dense_21/Tensordot/ShapeShape'token_and_position_embedding_14/add:z:0*
T0*
_output_shapes
:?
Ktransformer_block_3/multi_head_attention_3/dense_21/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ftransformer_block_3/multi_head_attention_3/dense_21/Tensordot/GatherV2GatherV2Ltransformer_block_3/multi_head_attention_3/dense_21/Tensordot/Shape:output:0Ktransformer_block_3/multi_head_attention_3/dense_21/Tensordot/free:output:0Ttransformer_block_3/multi_head_attention_3/dense_21/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Mtransformer_block_3/multi_head_attention_3/dense_21/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Htransformer_block_3/multi_head_attention_3/dense_21/Tensordot/GatherV2_1GatherV2Ltransformer_block_3/multi_head_attention_3/dense_21/Tensordot/Shape:output:0Ktransformer_block_3/multi_head_attention_3/dense_21/Tensordot/axes:output:0Vtransformer_block_3/multi_head_attention_3/dense_21/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Ctransformer_block_3/multi_head_attention_3/dense_21/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Btransformer_block_3/multi_head_attention_3/dense_21/Tensordot/ProdProdOtransformer_block_3/multi_head_attention_3/dense_21/Tensordot/GatherV2:output:0Ltransformer_block_3/multi_head_attention_3/dense_21/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
Etransformer_block_3/multi_head_attention_3/dense_21/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Dtransformer_block_3/multi_head_attention_3/dense_21/Tensordot/Prod_1ProdQtransformer_block_3/multi_head_attention_3/dense_21/Tensordot/GatherV2_1:output:0Ntransformer_block_3/multi_head_attention_3/dense_21/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
Itransformer_block_3/multi_head_attention_3/dense_21/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Dtransformer_block_3/multi_head_attention_3/dense_21/Tensordot/concatConcatV2Ktransformer_block_3/multi_head_attention_3/dense_21/Tensordot/free:output:0Ktransformer_block_3/multi_head_attention_3/dense_21/Tensordot/axes:output:0Rtransformer_block_3/multi_head_attention_3/dense_21/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Ctransformer_block_3/multi_head_attention_3/dense_21/Tensordot/stackPackKtransformer_block_3/multi_head_attention_3/dense_21/Tensordot/Prod:output:0Mtransformer_block_3/multi_head_attention_3/dense_21/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
Gtransformer_block_3/multi_head_attention_3/dense_21/Tensordot/transpose	Transpose'token_and_position_embedding_14/add:z:0Mtransformer_block_3/multi_head_attention_3/dense_21/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
Etransformer_block_3/multi_head_attention_3/dense_21/Tensordot/ReshapeReshapeKtransformer_block_3/multi_head_attention_3/dense_21/Tensordot/transpose:y:0Ltransformer_block_3/multi_head_attention_3/dense_21/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Dtransformer_block_3/multi_head_attention_3/dense_21/Tensordot/MatMulMatMulNtransformer_block_3/multi_head_attention_3/dense_21/Tensordot/Reshape:output:0Ttransformer_block_3/multi_head_attention_3/dense_21/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Etransformer_block_3/multi_head_attention_3/dense_21/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Ktransformer_block_3/multi_head_attention_3/dense_21/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ftransformer_block_3/multi_head_attention_3/dense_21/Tensordot/concat_1ConcatV2Otransformer_block_3/multi_head_attention_3/dense_21/Tensordot/GatherV2:output:0Ntransformer_block_3/multi_head_attention_3/dense_21/Tensordot/Const_2:output:0Ttransformer_block_3/multi_head_attention_3/dense_21/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
=transformer_block_3/multi_head_attention_3/dense_21/TensordotReshapeNtransformer_block_3/multi_head_attention_3/dense_21/Tensordot/MatMul:product:0Otransformer_block_3/multi_head_attention_3/dense_21/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
Jtransformer_block_3/multi_head_attention_3/dense_21/BiasAdd/ReadVariableOpReadVariableOpStransformer_block_3_multi_head_attention_3_dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
;transformer_block_3/multi_head_attention_3/dense_21/BiasAddBiasAddFtransformer_block_3/multi_head_attention_3/dense_21/Tensordot:output:0Rtransformer_block_3/multi_head_attention_3/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
Ltransformer_block_3/multi_head_attention_3/dense_22/Tensordot/ReadVariableOpReadVariableOpUtransformer_block_3_multi_head_attention_3_dense_22_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0?
Btransformer_block_3/multi_head_attention_3/dense_22/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
Btransformer_block_3/multi_head_attention_3/dense_22/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
Ctransformer_block_3/multi_head_attention_3/dense_22/Tensordot/ShapeShape'token_and_position_embedding_14/add:z:0*
T0*
_output_shapes
:?
Ktransformer_block_3/multi_head_attention_3/dense_22/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ftransformer_block_3/multi_head_attention_3/dense_22/Tensordot/GatherV2GatherV2Ltransformer_block_3/multi_head_attention_3/dense_22/Tensordot/Shape:output:0Ktransformer_block_3/multi_head_attention_3/dense_22/Tensordot/free:output:0Ttransformer_block_3/multi_head_attention_3/dense_22/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Mtransformer_block_3/multi_head_attention_3/dense_22/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Htransformer_block_3/multi_head_attention_3/dense_22/Tensordot/GatherV2_1GatherV2Ltransformer_block_3/multi_head_attention_3/dense_22/Tensordot/Shape:output:0Ktransformer_block_3/multi_head_attention_3/dense_22/Tensordot/axes:output:0Vtransformer_block_3/multi_head_attention_3/dense_22/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Ctransformer_block_3/multi_head_attention_3/dense_22/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Btransformer_block_3/multi_head_attention_3/dense_22/Tensordot/ProdProdOtransformer_block_3/multi_head_attention_3/dense_22/Tensordot/GatherV2:output:0Ltransformer_block_3/multi_head_attention_3/dense_22/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
Etransformer_block_3/multi_head_attention_3/dense_22/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Dtransformer_block_3/multi_head_attention_3/dense_22/Tensordot/Prod_1ProdQtransformer_block_3/multi_head_attention_3/dense_22/Tensordot/GatherV2_1:output:0Ntransformer_block_3/multi_head_attention_3/dense_22/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
Itransformer_block_3/multi_head_attention_3/dense_22/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Dtransformer_block_3/multi_head_attention_3/dense_22/Tensordot/concatConcatV2Ktransformer_block_3/multi_head_attention_3/dense_22/Tensordot/free:output:0Ktransformer_block_3/multi_head_attention_3/dense_22/Tensordot/axes:output:0Rtransformer_block_3/multi_head_attention_3/dense_22/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Ctransformer_block_3/multi_head_attention_3/dense_22/Tensordot/stackPackKtransformer_block_3/multi_head_attention_3/dense_22/Tensordot/Prod:output:0Mtransformer_block_3/multi_head_attention_3/dense_22/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
Gtransformer_block_3/multi_head_attention_3/dense_22/Tensordot/transpose	Transpose'token_and_position_embedding_14/add:z:0Mtransformer_block_3/multi_head_attention_3/dense_22/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
Etransformer_block_3/multi_head_attention_3/dense_22/Tensordot/ReshapeReshapeKtransformer_block_3/multi_head_attention_3/dense_22/Tensordot/transpose:y:0Ltransformer_block_3/multi_head_attention_3/dense_22/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Dtransformer_block_3/multi_head_attention_3/dense_22/Tensordot/MatMulMatMulNtransformer_block_3/multi_head_attention_3/dense_22/Tensordot/Reshape:output:0Ttransformer_block_3/multi_head_attention_3/dense_22/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Etransformer_block_3/multi_head_attention_3/dense_22/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Ktransformer_block_3/multi_head_attention_3/dense_22/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ftransformer_block_3/multi_head_attention_3/dense_22/Tensordot/concat_1ConcatV2Otransformer_block_3/multi_head_attention_3/dense_22/Tensordot/GatherV2:output:0Ntransformer_block_3/multi_head_attention_3/dense_22/Tensordot/Const_2:output:0Ttransformer_block_3/multi_head_attention_3/dense_22/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
=transformer_block_3/multi_head_attention_3/dense_22/TensordotReshapeNtransformer_block_3/multi_head_attention_3/dense_22/Tensordot/MatMul:product:0Otransformer_block_3/multi_head_attention_3/dense_22/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
Jtransformer_block_3/multi_head_attention_3/dense_22/BiasAdd/ReadVariableOpReadVariableOpStransformer_block_3_multi_head_attention_3_dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
;transformer_block_3/multi_head_attention_3/dense_22/BiasAddBiasAddFtransformer_block_3/multi_head_attention_3/dense_22/Tensordot:output:0Rtransformer_block_3/multi_head_attention_3/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
:transformer_block_3/multi_head_attention_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????|
:transformer_block_3/multi_head_attention_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :|
:transformer_block_3/multi_head_attention_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
8transformer_block_3/multi_head_attention_3/Reshape/shapePackAtransformer_block_3/multi_head_attention_3/strided_slice:output:0Ctransformer_block_3/multi_head_attention_3/Reshape/shape/1:output:0Ctransformer_block_3/multi_head_attention_3/Reshape/shape/2:output:0Ctransformer_block_3/multi_head_attention_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
2transformer_block_3/multi_head_attention_3/ReshapeReshapeDtransformer_block_3/multi_head_attention_3/dense_20/BiasAdd:output:0Atransformer_block_3/multi_head_attention_3/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"???????????????????
9transformer_block_3/multi_head_attention_3/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
4transformer_block_3/multi_head_attention_3/transpose	Transpose;transformer_block_3/multi_head_attention_3/Reshape:output:0Btransformer_block_3/multi_head_attention_3/transpose/perm:output:0*
T0*8
_output_shapes&
$:"???????????????????
<transformer_block_3/multi_head_attention_3/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????~
<transformer_block_3/multi_head_attention_3/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :~
<transformer_block_3/multi_head_attention_3/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
:transformer_block_3/multi_head_attention_3/Reshape_1/shapePackAtransformer_block_3/multi_head_attention_3/strided_slice:output:0Etransformer_block_3/multi_head_attention_3/Reshape_1/shape/1:output:0Etransformer_block_3/multi_head_attention_3/Reshape_1/shape/2:output:0Etransformer_block_3/multi_head_attention_3/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:?
4transformer_block_3/multi_head_attention_3/Reshape_1ReshapeDtransformer_block_3/multi_head_attention_3/dense_21/BiasAdd:output:0Ctransformer_block_3/multi_head_attention_3/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"???????????????????
;transformer_block_3/multi_head_attention_3/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
6transformer_block_3/multi_head_attention_3/transpose_1	Transpose=transformer_block_3/multi_head_attention_3/Reshape_1:output:0Dtransformer_block_3/multi_head_attention_3/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"???????????????????
<transformer_block_3/multi_head_attention_3/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????~
<transformer_block_3/multi_head_attention_3/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :~
<transformer_block_3/multi_head_attention_3/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
:transformer_block_3/multi_head_attention_3/Reshape_2/shapePackAtransformer_block_3/multi_head_attention_3/strided_slice:output:0Etransformer_block_3/multi_head_attention_3/Reshape_2/shape/1:output:0Etransformer_block_3/multi_head_attention_3/Reshape_2/shape/2:output:0Etransformer_block_3/multi_head_attention_3/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:?
4transformer_block_3/multi_head_attention_3/Reshape_2ReshapeDtransformer_block_3/multi_head_attention_3/dense_22/BiasAdd:output:0Ctransformer_block_3/multi_head_attention_3/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"???????????????????
;transformer_block_3/multi_head_attention_3/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
6transformer_block_3/multi_head_attention_3/transpose_2	Transpose=transformer_block_3/multi_head_attention_3/Reshape_2:output:0Dtransformer_block_3/multi_head_attention_3/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"???????????????????
1transformer_block_3/multi_head_attention_3/MatMulBatchMatMulV28transformer_block_3/multi_head_attention_3/transpose:y:0:transformer_block_3/multi_head_attention_3/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(?
2transformer_block_3/multi_head_attention_3/Shape_1Shape:transformer_block_3/multi_head_attention_3/transpose_1:y:0*
T0*
_output_shapes
:?
@transformer_block_3/multi_head_attention_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Btransformer_block_3/multi_head_attention_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Btransformer_block_3/multi_head_attention_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
:transformer_block_3/multi_head_attention_3/strided_slice_1StridedSlice;transformer_block_3/multi_head_attention_3/Shape_1:output:0Itransformer_block_3/multi_head_attention_3/strided_slice_1/stack:output:0Ktransformer_block_3/multi_head_attention_3/strided_slice_1/stack_1:output:0Ktransformer_block_3/multi_head_attention_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
/transformer_block_3/multi_head_attention_3/CastCastCtransformer_block_3/multi_head_attention_3/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
/transformer_block_3/multi_head_attention_3/SqrtSqrt3transformer_block_3/multi_head_attention_3/Cast:y:0*
T0*
_output_shapes
: ?
2transformer_block_3/multi_head_attention_3/truedivRealDiv:transformer_block_3/multi_head_attention_3/MatMul:output:03transformer_block_3/multi_head_attention_3/Sqrt:y:0*
T0*A
_output_shapes/
-:+????????????????????????????
2transformer_block_3/multi_head_attention_3/SoftmaxSoftmax6transformer_block_3/multi_head_attention_3/truediv:z:0*
T0*A
_output_shapes/
-:+????????????????????????????
3transformer_block_3/multi_head_attention_3/MatMul_1BatchMatMulV2<transformer_block_3/multi_head_attention_3/Softmax:softmax:0:transformer_block_3/multi_head_attention_3/transpose_2:y:0*
T0*8
_output_shapes&
$:"???????????????????
;transformer_block_3/multi_head_attention_3/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
6transformer_block_3/multi_head_attention_3/transpose_3	Transpose<transformer_block_3/multi_head_attention_3/MatMul_1:output:0Dtransformer_block_3/multi_head_attention_3/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"???????????????????
<transformer_block_3/multi_head_attention_3/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????~
<transformer_block_3/multi_head_attention_3/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
:transformer_block_3/multi_head_attention_3/Reshape_3/shapePackAtransformer_block_3/multi_head_attention_3/strided_slice:output:0Etransformer_block_3/multi_head_attention_3/Reshape_3/shape/1:output:0Etransformer_block_3/multi_head_attention_3/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:?
4transformer_block_3/multi_head_attention_3/Reshape_3Reshape:transformer_block_3/multi_head_attention_3/transpose_3:y:0Ctransformer_block_3/multi_head_attention_3/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :???????????????????
Ltransformer_block_3/multi_head_attention_3/dense_23/Tensordot/ReadVariableOpReadVariableOpUtransformer_block_3_multi_head_attention_3_dense_23_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0?
Btransformer_block_3/multi_head_attention_3/dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
Btransformer_block_3/multi_head_attention_3/dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
Ctransformer_block_3/multi_head_attention_3/dense_23/Tensordot/ShapeShape=transformer_block_3/multi_head_attention_3/Reshape_3:output:0*
T0*
_output_shapes
:?
Ktransformer_block_3/multi_head_attention_3/dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ftransformer_block_3/multi_head_attention_3/dense_23/Tensordot/GatherV2GatherV2Ltransformer_block_3/multi_head_attention_3/dense_23/Tensordot/Shape:output:0Ktransformer_block_3/multi_head_attention_3/dense_23/Tensordot/free:output:0Ttransformer_block_3/multi_head_attention_3/dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Mtransformer_block_3/multi_head_attention_3/dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Htransformer_block_3/multi_head_attention_3/dense_23/Tensordot/GatherV2_1GatherV2Ltransformer_block_3/multi_head_attention_3/dense_23/Tensordot/Shape:output:0Ktransformer_block_3/multi_head_attention_3/dense_23/Tensordot/axes:output:0Vtransformer_block_3/multi_head_attention_3/dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Ctransformer_block_3/multi_head_attention_3/dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Btransformer_block_3/multi_head_attention_3/dense_23/Tensordot/ProdProdOtransformer_block_3/multi_head_attention_3/dense_23/Tensordot/GatherV2:output:0Ltransformer_block_3/multi_head_attention_3/dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
Etransformer_block_3/multi_head_attention_3/dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Dtransformer_block_3/multi_head_attention_3/dense_23/Tensordot/Prod_1ProdQtransformer_block_3/multi_head_attention_3/dense_23/Tensordot/GatherV2_1:output:0Ntransformer_block_3/multi_head_attention_3/dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
Itransformer_block_3/multi_head_attention_3/dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Dtransformer_block_3/multi_head_attention_3/dense_23/Tensordot/concatConcatV2Ktransformer_block_3/multi_head_attention_3/dense_23/Tensordot/free:output:0Ktransformer_block_3/multi_head_attention_3/dense_23/Tensordot/axes:output:0Rtransformer_block_3/multi_head_attention_3/dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Ctransformer_block_3/multi_head_attention_3/dense_23/Tensordot/stackPackKtransformer_block_3/multi_head_attention_3/dense_23/Tensordot/Prod:output:0Mtransformer_block_3/multi_head_attention_3/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
Gtransformer_block_3/multi_head_attention_3/dense_23/Tensordot/transpose	Transpose=transformer_block_3/multi_head_attention_3/Reshape_3:output:0Mtransformer_block_3/multi_head_attention_3/dense_23/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :???????????????????
Etransformer_block_3/multi_head_attention_3/dense_23/Tensordot/ReshapeReshapeKtransformer_block_3/multi_head_attention_3/dense_23/Tensordot/transpose:y:0Ltransformer_block_3/multi_head_attention_3/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Dtransformer_block_3/multi_head_attention_3/dense_23/Tensordot/MatMulMatMulNtransformer_block_3/multi_head_attention_3/dense_23/Tensordot/Reshape:output:0Ttransformer_block_3/multi_head_attention_3/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Etransformer_block_3/multi_head_attention_3/dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Ktransformer_block_3/multi_head_attention_3/dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ftransformer_block_3/multi_head_attention_3/dense_23/Tensordot/concat_1ConcatV2Otransformer_block_3/multi_head_attention_3/dense_23/Tensordot/GatherV2:output:0Ntransformer_block_3/multi_head_attention_3/dense_23/Tensordot/Const_2:output:0Ttransformer_block_3/multi_head_attention_3/dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
=transformer_block_3/multi_head_attention_3/dense_23/TensordotReshapeNtransformer_block_3/multi_head_attention_3/dense_23/Tensordot/MatMul:product:0Otransformer_block_3/multi_head_attention_3/dense_23/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :???????????????????
Jtransformer_block_3/multi_head_attention_3/dense_23/BiasAdd/ReadVariableOpReadVariableOpStransformer_block_3_multi_head_attention_3_dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
;transformer_block_3/multi_head_attention_3/dense_23/BiasAddBiasAddFtransformer_block_3/multi_head_attention_3/dense_23/Tensordot:output:0Rtransformer_block_3/multi_head_attention_3/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :???????????????????
&transformer_block_3/dropout_8/IdentityIdentityDtransformer_block_3/multi_head_attention_3/dense_23/BiasAdd:output:0*
T0*4
_output_shapes"
 :???????????????????
transformer_block_3/addAddV2'token_and_position_embedding_14/add:z:0/transformer_block_3/dropout_8/Identity:output:0*
T0*,
_output_shapes
:??????????'?
Htransformer_block_3/layer_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
6transformer_block_3/layer_normalization_6/moments/meanMeantransformer_block_3/add:z:0Qtransformer_block_3/layer_normalization_6/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????'*
	keep_dims(?
>transformer_block_3/layer_normalization_6/moments/StopGradientStopGradient?transformer_block_3/layer_normalization_6/moments/mean:output:0*
T0*,
_output_shapes
:??????????'?
Ctransformer_block_3/layer_normalization_6/moments/SquaredDifferenceSquaredDifferencetransformer_block_3/add:z:0Gtransformer_block_3/layer_normalization_6/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????'?
Ltransformer_block_3/layer_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
:transformer_block_3/layer_normalization_6/moments/varianceMeanGtransformer_block_3/layer_normalization_6/moments/SquaredDifference:z:0Utransformer_block_3/layer_normalization_6/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????'*
	keep_dims(~
9transformer_block_3/layer_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?5?
7transformer_block_3/layer_normalization_6/batchnorm/addAddV2Ctransformer_block_3/layer_normalization_6/moments/variance:output:0Btransformer_block_3/layer_normalization_6/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????'?
9transformer_block_3/layer_normalization_6/batchnorm/RsqrtRsqrt;transformer_block_3/layer_normalization_6/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????'?
Ftransformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_3_layer_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
7transformer_block_3/layer_normalization_6/batchnorm/mulMul=transformer_block_3/layer_normalization_6/batchnorm/Rsqrt:y:0Ntransformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
9transformer_block_3/layer_normalization_6/batchnorm/mul_1Multransformer_block_3/add:z:0;transformer_block_3/layer_normalization_6/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????'?
9transformer_block_3/layer_normalization_6/batchnorm/mul_2Mul?transformer_block_3/layer_normalization_6/moments/mean:output:0;transformer_block_3/layer_normalization_6/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????'?
Btransformer_block_3/layer_normalization_6/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_3_layer_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
7transformer_block_3/layer_normalization_6/batchnorm/subSubJtransformer_block_3/layer_normalization_6/batchnorm/ReadVariableOp:value:0=transformer_block_3/layer_normalization_6/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????'?
9transformer_block_3/layer_normalization_6/batchnorm/add_1AddV2=transformer_block_3/layer_normalization_6/batchnorm/mul_1:z:0;transformer_block_3/layer_normalization_6/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????'?
Btransformer_block_3/sequential_3/dense_24/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_3_sequential_3_dense_24_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0?
8transformer_block_3/sequential_3/dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
8transformer_block_3/sequential_3/dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
9transformer_block_3/sequential_3/dense_24/Tensordot/ShapeShape=transformer_block_3/layer_normalization_6/batchnorm/add_1:z:0*
T0*
_output_shapes
:?
Atransformer_block_3/sequential_3/dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
<transformer_block_3/sequential_3/dense_24/Tensordot/GatherV2GatherV2Btransformer_block_3/sequential_3/dense_24/Tensordot/Shape:output:0Atransformer_block_3/sequential_3/dense_24/Tensordot/free:output:0Jtransformer_block_3/sequential_3/dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Ctransformer_block_3/sequential_3/dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
>transformer_block_3/sequential_3/dense_24/Tensordot/GatherV2_1GatherV2Btransformer_block_3/sequential_3/dense_24/Tensordot/Shape:output:0Atransformer_block_3/sequential_3/dense_24/Tensordot/axes:output:0Ltransformer_block_3/sequential_3/dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
9transformer_block_3/sequential_3/dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
8transformer_block_3/sequential_3/dense_24/Tensordot/ProdProdEtransformer_block_3/sequential_3/dense_24/Tensordot/GatherV2:output:0Btransformer_block_3/sequential_3/dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
;transformer_block_3/sequential_3/dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
:transformer_block_3/sequential_3/dense_24/Tensordot/Prod_1ProdGtransformer_block_3/sequential_3/dense_24/Tensordot/GatherV2_1:output:0Dtransformer_block_3/sequential_3/dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
?transformer_block_3/sequential_3/dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
:transformer_block_3/sequential_3/dense_24/Tensordot/concatConcatV2Atransformer_block_3/sequential_3/dense_24/Tensordot/free:output:0Atransformer_block_3/sequential_3/dense_24/Tensordot/axes:output:0Htransformer_block_3/sequential_3/dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
9transformer_block_3/sequential_3/dense_24/Tensordot/stackPackAtransformer_block_3/sequential_3/dense_24/Tensordot/Prod:output:0Ctransformer_block_3/sequential_3/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
=transformer_block_3/sequential_3/dense_24/Tensordot/transpose	Transpose=transformer_block_3/layer_normalization_6/batchnorm/add_1:z:0Ctransformer_block_3/sequential_3/dense_24/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
;transformer_block_3/sequential_3/dense_24/Tensordot/ReshapeReshapeAtransformer_block_3/sequential_3/dense_24/Tensordot/transpose:y:0Btransformer_block_3/sequential_3/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
:transformer_block_3/sequential_3/dense_24/Tensordot/MatMulMatMulDtransformer_block_3/sequential_3/dense_24/Tensordot/Reshape:output:0Jtransformer_block_3/sequential_3/dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
;transformer_block_3/sequential_3/dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Atransformer_block_3/sequential_3/dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
<transformer_block_3/sequential_3/dense_24/Tensordot/concat_1ConcatV2Etransformer_block_3/sequential_3/dense_24/Tensordot/GatherV2:output:0Dtransformer_block_3/sequential_3/dense_24/Tensordot/Const_2:output:0Jtransformer_block_3/sequential_3/dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
3transformer_block_3/sequential_3/dense_24/TensordotReshapeDtransformer_block_3/sequential_3/dense_24/Tensordot/MatMul:product:0Etransformer_block_3/sequential_3/dense_24/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
@transformer_block_3/sequential_3/dense_24/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_3_sequential_3_dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
1transformer_block_3/sequential_3/dense_24/BiasAddBiasAdd<transformer_block_3/sequential_3/dense_24/Tensordot:output:0Htransformer_block_3/sequential_3/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
.transformer_block_3/sequential_3/dense_24/ReluRelu:transformer_block_3/sequential_3/dense_24/BiasAdd:output:0*
T0*,
_output_shapes
:??????????'?
Btransformer_block_3/sequential_3/dense_25/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_3_sequential_3_dense_25_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0?
8transformer_block_3/sequential_3/dense_25/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
8transformer_block_3/sequential_3/dense_25/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
9transformer_block_3/sequential_3/dense_25/Tensordot/ShapeShape<transformer_block_3/sequential_3/dense_24/Relu:activations:0*
T0*
_output_shapes
:?
Atransformer_block_3/sequential_3/dense_25/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
<transformer_block_3/sequential_3/dense_25/Tensordot/GatherV2GatherV2Btransformer_block_3/sequential_3/dense_25/Tensordot/Shape:output:0Atransformer_block_3/sequential_3/dense_25/Tensordot/free:output:0Jtransformer_block_3/sequential_3/dense_25/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Ctransformer_block_3/sequential_3/dense_25/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
>transformer_block_3/sequential_3/dense_25/Tensordot/GatherV2_1GatherV2Btransformer_block_3/sequential_3/dense_25/Tensordot/Shape:output:0Atransformer_block_3/sequential_3/dense_25/Tensordot/axes:output:0Ltransformer_block_3/sequential_3/dense_25/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
9transformer_block_3/sequential_3/dense_25/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
8transformer_block_3/sequential_3/dense_25/Tensordot/ProdProdEtransformer_block_3/sequential_3/dense_25/Tensordot/GatherV2:output:0Btransformer_block_3/sequential_3/dense_25/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
;transformer_block_3/sequential_3/dense_25/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
:transformer_block_3/sequential_3/dense_25/Tensordot/Prod_1ProdGtransformer_block_3/sequential_3/dense_25/Tensordot/GatherV2_1:output:0Dtransformer_block_3/sequential_3/dense_25/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
?transformer_block_3/sequential_3/dense_25/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
:transformer_block_3/sequential_3/dense_25/Tensordot/concatConcatV2Atransformer_block_3/sequential_3/dense_25/Tensordot/free:output:0Atransformer_block_3/sequential_3/dense_25/Tensordot/axes:output:0Htransformer_block_3/sequential_3/dense_25/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
9transformer_block_3/sequential_3/dense_25/Tensordot/stackPackAtransformer_block_3/sequential_3/dense_25/Tensordot/Prod:output:0Ctransformer_block_3/sequential_3/dense_25/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
=transformer_block_3/sequential_3/dense_25/Tensordot/transpose	Transpose<transformer_block_3/sequential_3/dense_24/Relu:activations:0Ctransformer_block_3/sequential_3/dense_25/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
;transformer_block_3/sequential_3/dense_25/Tensordot/ReshapeReshapeAtransformer_block_3/sequential_3/dense_25/Tensordot/transpose:y:0Btransformer_block_3/sequential_3/dense_25/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
:transformer_block_3/sequential_3/dense_25/Tensordot/MatMulMatMulDtransformer_block_3/sequential_3/dense_25/Tensordot/Reshape:output:0Jtransformer_block_3/sequential_3/dense_25/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
;transformer_block_3/sequential_3/dense_25/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
Atransformer_block_3/sequential_3/dense_25/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
<transformer_block_3/sequential_3/dense_25/Tensordot/concat_1ConcatV2Etransformer_block_3/sequential_3/dense_25/Tensordot/GatherV2:output:0Dtransformer_block_3/sequential_3/dense_25/Tensordot/Const_2:output:0Jtransformer_block_3/sequential_3/dense_25/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
3transformer_block_3/sequential_3/dense_25/TensordotReshapeDtransformer_block_3/sequential_3/dense_25/Tensordot/MatMul:product:0Etransformer_block_3/sequential_3/dense_25/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
@transformer_block_3/sequential_3/dense_25/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_3_sequential_3_dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
1transformer_block_3/sequential_3/dense_25/BiasAddBiasAdd<transformer_block_3/sequential_3/dense_25/Tensordot:output:0Htransformer_block_3/sequential_3/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
&transformer_block_3/dropout_9/IdentityIdentity:transformer_block_3/sequential_3/dense_25/BiasAdd:output:0*
T0*,
_output_shapes
:??????????'?
transformer_block_3/add_1AddV2=transformer_block_3/layer_normalization_6/batchnorm/add_1:z:0/transformer_block_3/dropout_9/Identity:output:0*
T0*,
_output_shapes
:??????????'?
Htransformer_block_3/layer_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
6transformer_block_3/layer_normalization_7/moments/meanMeantransformer_block_3/add_1:z:0Qtransformer_block_3/layer_normalization_7/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????'*
	keep_dims(?
>transformer_block_3/layer_normalization_7/moments/StopGradientStopGradient?transformer_block_3/layer_normalization_7/moments/mean:output:0*
T0*,
_output_shapes
:??????????'?
Ctransformer_block_3/layer_normalization_7/moments/SquaredDifferenceSquaredDifferencetransformer_block_3/add_1:z:0Gtransformer_block_3/layer_normalization_7/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????'?
Ltransformer_block_3/layer_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
:transformer_block_3/layer_normalization_7/moments/varianceMeanGtransformer_block_3/layer_normalization_7/moments/SquaredDifference:z:0Utransformer_block_3/layer_normalization_7/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????'*
	keep_dims(~
9transformer_block_3/layer_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?5?
7transformer_block_3/layer_normalization_7/batchnorm/addAddV2Ctransformer_block_3/layer_normalization_7/moments/variance:output:0Btransformer_block_3/layer_normalization_7/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????'?
9transformer_block_3/layer_normalization_7/batchnorm/RsqrtRsqrt;transformer_block_3/layer_normalization_7/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????'?
Ftransformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_3_layer_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
7transformer_block_3/layer_normalization_7/batchnorm/mulMul=transformer_block_3/layer_normalization_7/batchnorm/Rsqrt:y:0Ntransformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
9transformer_block_3/layer_normalization_7/batchnorm/mul_1Multransformer_block_3/add_1:z:0;transformer_block_3/layer_normalization_7/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????'?
9transformer_block_3/layer_normalization_7/batchnorm/mul_2Mul?transformer_block_3/layer_normalization_7/moments/mean:output:0;transformer_block_3/layer_normalization_7/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????'?
Btransformer_block_3/layer_normalization_7/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_3_layer_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
7transformer_block_3/layer_normalization_7/batchnorm/subSubJtransformer_block_3/layer_normalization_7/batchnorm/ReadVariableOp:value:0=transformer_block_3/layer_normalization_7/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????'?
9transformer_block_3/layer_normalization_7/batchnorm/add_1AddV2=transformer_block_3/layer_normalization_7/batchnorm/mul_1:z:0;transformer_block_3/layer_normalization_7/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????'s
1global_average_pooling1d_3/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
global_average_pooling1d_3/MeanMean=transformer_block_3/layer_normalization_7/batchnorm/add_1:z:0:global_average_pooling1d_3/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????{
dropout_10/IdentityIdentity(global_average_pooling1d_3/Mean:output:0*
T0*'
_output_shapes
:??????????
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_26/MatMulMatMuldropout_10/Identity:output:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*'
_output_shapes
:?????????n
dropout_11/IdentityIdentitydense_26/Relu:activations:0*
T0*'
_output_shapes
:??????????
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_27/MatMulMatMuldropout_11/Identity:output:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_27/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp>^token_and_position_embedding_14/embedding_32/embedding_lookup>^token_and_position_embedding_14/embedding_33/embedding_lookupC^transformer_block_3/layer_normalization_6/batchnorm/ReadVariableOpG^transformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOpC^transformer_block_3/layer_normalization_7/batchnorm/ReadVariableOpG^transformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOpK^transformer_block_3/multi_head_attention_3/dense_20/BiasAdd/ReadVariableOpM^transformer_block_3/multi_head_attention_3/dense_20/Tensordot/ReadVariableOpK^transformer_block_3/multi_head_attention_3/dense_21/BiasAdd/ReadVariableOpM^transformer_block_3/multi_head_attention_3/dense_21/Tensordot/ReadVariableOpK^transformer_block_3/multi_head_attention_3/dense_22/BiasAdd/ReadVariableOpM^transformer_block_3/multi_head_attention_3/dense_22/Tensordot/ReadVariableOpK^transformer_block_3/multi_head_attention_3/dense_23/BiasAdd/ReadVariableOpM^transformer_block_3/multi_head_attention_3/dense_23/Tensordot/ReadVariableOpA^transformer_block_3/sequential_3/dense_24/BiasAdd/ReadVariableOpC^transformer_block_3/sequential_3/dense_24/Tensordot/ReadVariableOpA^transformer_block_3/sequential_3/dense_25/BiasAdd/ReadVariableOpC^transformer_block_3/sequential_3/dense_25/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:??????????': : : : : : : : : : : : : : : : : : : : : : 2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2~
=token_and_position_embedding_14/embedding_32/embedding_lookup=token_and_position_embedding_14/embedding_32/embedding_lookup2~
=token_and_position_embedding_14/embedding_33/embedding_lookup=token_and_position_embedding_14/embedding_33/embedding_lookup2?
Btransformer_block_3/layer_normalization_6/batchnorm/ReadVariableOpBtransformer_block_3/layer_normalization_6/batchnorm/ReadVariableOp2?
Ftransformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOpFtransformer_block_3/layer_normalization_6/batchnorm/mul/ReadVariableOp2?
Btransformer_block_3/layer_normalization_7/batchnorm/ReadVariableOpBtransformer_block_3/layer_normalization_7/batchnorm/ReadVariableOp2?
Ftransformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOpFtransformer_block_3/layer_normalization_7/batchnorm/mul/ReadVariableOp2?
Jtransformer_block_3/multi_head_attention_3/dense_20/BiasAdd/ReadVariableOpJtransformer_block_3/multi_head_attention_3/dense_20/BiasAdd/ReadVariableOp2?
Ltransformer_block_3/multi_head_attention_3/dense_20/Tensordot/ReadVariableOpLtransformer_block_3/multi_head_attention_3/dense_20/Tensordot/ReadVariableOp2?
Jtransformer_block_3/multi_head_attention_3/dense_21/BiasAdd/ReadVariableOpJtransformer_block_3/multi_head_attention_3/dense_21/BiasAdd/ReadVariableOp2?
Ltransformer_block_3/multi_head_attention_3/dense_21/Tensordot/ReadVariableOpLtransformer_block_3/multi_head_attention_3/dense_21/Tensordot/ReadVariableOp2?
Jtransformer_block_3/multi_head_attention_3/dense_22/BiasAdd/ReadVariableOpJtransformer_block_3/multi_head_attention_3/dense_22/BiasAdd/ReadVariableOp2?
Ltransformer_block_3/multi_head_attention_3/dense_22/Tensordot/ReadVariableOpLtransformer_block_3/multi_head_attention_3/dense_22/Tensordot/ReadVariableOp2?
Jtransformer_block_3/multi_head_attention_3/dense_23/BiasAdd/ReadVariableOpJtransformer_block_3/multi_head_attention_3/dense_23/BiasAdd/ReadVariableOp2?
Ltransformer_block_3/multi_head_attention_3/dense_23/Tensordot/ReadVariableOpLtransformer_block_3/multi_head_attention_3/dense_23/Tensordot/ReadVariableOp2?
@transformer_block_3/sequential_3/dense_24/BiasAdd/ReadVariableOp@transformer_block_3/sequential_3/dense_24/BiasAdd/ReadVariableOp2?
Btransformer_block_3/sequential_3/dense_24/Tensordot/ReadVariableOpBtransformer_block_3/sequential_3/dense_24/Tensordot/ReadVariableOp2?
@transformer_block_3/sequential_3/dense_25/BiasAdd/ReadVariableOp@transformer_block_3/sequential_3/dense_25/BiasAdd/ReadVariableOp2?
Btransformer_block_3/sequential_3/dense_25/Tensordot/ReadVariableOpBtransformer_block_3/sequential_3/dense_25/Tensordot/ReadVariableOp:T P
,
_output_shapes
:??????????'
 
_user_specified_nameinputs
?
c
*__inference_dropout_10_layer_call_fn_19525

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_10_layer_call_and_return_conditional_losses_17461o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
V
:__inference_global_average_pooling1d_3_layer_call_fn_19498

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_global_average_pooling1d_3_layer_call_and_return_conditional_losses_16973i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
c
*__inference_dropout_11_layer_call_fn_19572

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_17428o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_3_layer_call_and_return_conditional_losses_16949
dense_24_input 
dense_24_16938:
dense_24_16940: 
dense_25_16943:
dense_25_16945:
identity?? dense_24/StatefulPartitionedCall? dense_25/StatefulPartitionedCall?
 dense_24/StatefulPartitionedCallStatefulPartitionedCalldense_24_inputdense_24_16938dense_24_16940*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_16808?
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_16943dense_25_16945*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_25_layer_call_and_return_conditional_losses_16844}
IdentityIdentity)dense_25/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????'?
NoOpNoOp!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????': : : : 2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall:\ X
,
_output_shapes
:??????????'
(
_user_specified_namedense_24_input
?
?
C__inference_dense_24_layer_call_and_return_conditional_losses_19788

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????'f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:??????????'z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????': : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????'
 
_user_specified_nameinputs
??
?
N__inference_transformer_block_3_layer_call_and_return_conditional_losses_19493

inputsS
Amulti_head_attention_3_dense_20_tensordot_readvariableop_resource:M
?multi_head_attention_3_dense_20_biasadd_readvariableop_resource:S
Amulti_head_attention_3_dense_21_tensordot_readvariableop_resource:M
?multi_head_attention_3_dense_21_biasadd_readvariableop_resource:S
Amulti_head_attention_3_dense_22_tensordot_readvariableop_resource:M
?multi_head_attention_3_dense_22_biasadd_readvariableop_resource:S
Amulti_head_attention_3_dense_23_tensordot_readvariableop_resource:M
?multi_head_attention_3_dense_23_biasadd_readvariableop_resource:I
;layer_normalization_6_batchnorm_mul_readvariableop_resource:E
7layer_normalization_6_batchnorm_readvariableop_resource:I
7sequential_3_dense_24_tensordot_readvariableop_resource:C
5sequential_3_dense_24_biasadd_readvariableop_resource:I
7sequential_3_dense_25_tensordot_readvariableop_resource:C
5sequential_3_dense_25_biasadd_readvariableop_resource:I
;layer_normalization_7_batchnorm_mul_readvariableop_resource:E
7layer_normalization_7_batchnorm_readvariableop_resource:
identity??.layer_normalization_6/batchnorm/ReadVariableOp?2layer_normalization_6/batchnorm/mul/ReadVariableOp?.layer_normalization_7/batchnorm/ReadVariableOp?2layer_normalization_7/batchnorm/mul/ReadVariableOp?6multi_head_attention_3/dense_20/BiasAdd/ReadVariableOp?8multi_head_attention_3/dense_20/Tensordot/ReadVariableOp?6multi_head_attention_3/dense_21/BiasAdd/ReadVariableOp?8multi_head_attention_3/dense_21/Tensordot/ReadVariableOp?6multi_head_attention_3/dense_22/BiasAdd/ReadVariableOp?8multi_head_attention_3/dense_22/Tensordot/ReadVariableOp?6multi_head_attention_3/dense_23/BiasAdd/ReadVariableOp?8multi_head_attention_3/dense_23/Tensordot/ReadVariableOp?,sequential_3/dense_24/BiasAdd/ReadVariableOp?.sequential_3/dense_24/Tensordot/ReadVariableOp?,sequential_3/dense_25/BiasAdd/ReadVariableOp?.sequential_3/dense_25/Tensordot/ReadVariableOpR
multi_head_attention_3/ShapeShapeinputs*
T0*
_output_shapes
:t
*multi_head_attention_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,multi_head_attention_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,multi_head_attention_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$multi_head_attention_3/strided_sliceStridedSlice%multi_head_attention_3/Shape:output:03multi_head_attention_3/strided_slice/stack:output:05multi_head_attention_3/strided_slice/stack_1:output:05multi_head_attention_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
8multi_head_attention_3/dense_20/Tensordot/ReadVariableOpReadVariableOpAmulti_head_attention_3_dense_20_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0x
.multi_head_attention_3/dense_20/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
.multi_head_attention_3/dense_20/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       e
/multi_head_attention_3/dense_20/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:y
7multi_head_attention_3/dense_20/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_attention_3/dense_20/Tensordot/GatherV2GatherV28multi_head_attention_3/dense_20/Tensordot/Shape:output:07multi_head_attention_3/dense_20/Tensordot/free:output:0@multi_head_attention_3/dense_20/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9multi_head_attention_3/dense_20/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_attention_3/dense_20/Tensordot/GatherV2_1GatherV28multi_head_attention_3/dense_20/Tensordot/Shape:output:07multi_head_attention_3/dense_20/Tensordot/axes:output:0Bmulti_head_attention_3/dense_20/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/multi_head_attention_3/dense_20/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.multi_head_attention_3/dense_20/Tensordot/ProdProd;multi_head_attention_3/dense_20/Tensordot/GatherV2:output:08multi_head_attention_3/dense_20/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_attention_3/dense_20/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
0multi_head_attention_3/dense_20/Tensordot/Prod_1Prod=multi_head_attention_3/dense_20/Tensordot/GatherV2_1:output:0:multi_head_attention_3/dense_20/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5multi_head_attention_3/dense_20/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0multi_head_attention_3/dense_20/Tensordot/concatConcatV27multi_head_attention_3/dense_20/Tensordot/free:output:07multi_head_attention_3/dense_20/Tensordot/axes:output:0>multi_head_attention_3/dense_20/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
/multi_head_attention_3/dense_20/Tensordot/stackPack7multi_head_attention_3/dense_20/Tensordot/Prod:output:09multi_head_attention_3/dense_20/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
3multi_head_attention_3/dense_20/Tensordot/transpose	Transposeinputs9multi_head_attention_3/dense_20/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
1multi_head_attention_3/dense_20/Tensordot/ReshapeReshape7multi_head_attention_3/dense_20/Tensordot/transpose:y:08multi_head_attention_3/dense_20/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
0multi_head_attention_3/dense_20/Tensordot/MatMulMatMul:multi_head_attention_3/dense_20/Tensordot/Reshape:output:0@multi_head_attention_3/dense_20/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????{
1multi_head_attention_3/dense_20/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:y
7multi_head_attention_3/dense_20/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_attention_3/dense_20/Tensordot/concat_1ConcatV2;multi_head_attention_3/dense_20/Tensordot/GatherV2:output:0:multi_head_attention_3/dense_20/Tensordot/Const_2:output:0@multi_head_attention_3/dense_20/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
)multi_head_attention_3/dense_20/TensordotReshape:multi_head_attention_3/dense_20/Tensordot/MatMul:product:0;multi_head_attention_3/dense_20/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
6multi_head_attention_3/dense_20/BiasAdd/ReadVariableOpReadVariableOp?multi_head_attention_3_dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'multi_head_attention_3/dense_20/BiasAddBiasAdd2multi_head_attention_3/dense_20/Tensordot:output:0>multi_head_attention_3/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
8multi_head_attention_3/dense_21/Tensordot/ReadVariableOpReadVariableOpAmulti_head_attention_3_dense_21_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0x
.multi_head_attention_3/dense_21/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
.multi_head_attention_3/dense_21/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       e
/multi_head_attention_3/dense_21/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:y
7multi_head_attention_3/dense_21/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_attention_3/dense_21/Tensordot/GatherV2GatherV28multi_head_attention_3/dense_21/Tensordot/Shape:output:07multi_head_attention_3/dense_21/Tensordot/free:output:0@multi_head_attention_3/dense_21/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9multi_head_attention_3/dense_21/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_attention_3/dense_21/Tensordot/GatherV2_1GatherV28multi_head_attention_3/dense_21/Tensordot/Shape:output:07multi_head_attention_3/dense_21/Tensordot/axes:output:0Bmulti_head_attention_3/dense_21/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/multi_head_attention_3/dense_21/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.multi_head_attention_3/dense_21/Tensordot/ProdProd;multi_head_attention_3/dense_21/Tensordot/GatherV2:output:08multi_head_attention_3/dense_21/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_attention_3/dense_21/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
0multi_head_attention_3/dense_21/Tensordot/Prod_1Prod=multi_head_attention_3/dense_21/Tensordot/GatherV2_1:output:0:multi_head_attention_3/dense_21/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5multi_head_attention_3/dense_21/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0multi_head_attention_3/dense_21/Tensordot/concatConcatV27multi_head_attention_3/dense_21/Tensordot/free:output:07multi_head_attention_3/dense_21/Tensordot/axes:output:0>multi_head_attention_3/dense_21/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
/multi_head_attention_3/dense_21/Tensordot/stackPack7multi_head_attention_3/dense_21/Tensordot/Prod:output:09multi_head_attention_3/dense_21/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
3multi_head_attention_3/dense_21/Tensordot/transpose	Transposeinputs9multi_head_attention_3/dense_21/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
1multi_head_attention_3/dense_21/Tensordot/ReshapeReshape7multi_head_attention_3/dense_21/Tensordot/transpose:y:08multi_head_attention_3/dense_21/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
0multi_head_attention_3/dense_21/Tensordot/MatMulMatMul:multi_head_attention_3/dense_21/Tensordot/Reshape:output:0@multi_head_attention_3/dense_21/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????{
1multi_head_attention_3/dense_21/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:y
7multi_head_attention_3/dense_21/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_attention_3/dense_21/Tensordot/concat_1ConcatV2;multi_head_attention_3/dense_21/Tensordot/GatherV2:output:0:multi_head_attention_3/dense_21/Tensordot/Const_2:output:0@multi_head_attention_3/dense_21/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
)multi_head_attention_3/dense_21/TensordotReshape:multi_head_attention_3/dense_21/Tensordot/MatMul:product:0;multi_head_attention_3/dense_21/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
6multi_head_attention_3/dense_21/BiasAdd/ReadVariableOpReadVariableOp?multi_head_attention_3_dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'multi_head_attention_3/dense_21/BiasAddBiasAdd2multi_head_attention_3/dense_21/Tensordot:output:0>multi_head_attention_3/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
8multi_head_attention_3/dense_22/Tensordot/ReadVariableOpReadVariableOpAmulti_head_attention_3_dense_22_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0x
.multi_head_attention_3/dense_22/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
.multi_head_attention_3/dense_22/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       e
/multi_head_attention_3/dense_22/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:y
7multi_head_attention_3/dense_22/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_attention_3/dense_22/Tensordot/GatherV2GatherV28multi_head_attention_3/dense_22/Tensordot/Shape:output:07multi_head_attention_3/dense_22/Tensordot/free:output:0@multi_head_attention_3/dense_22/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9multi_head_attention_3/dense_22/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_attention_3/dense_22/Tensordot/GatherV2_1GatherV28multi_head_attention_3/dense_22/Tensordot/Shape:output:07multi_head_attention_3/dense_22/Tensordot/axes:output:0Bmulti_head_attention_3/dense_22/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/multi_head_attention_3/dense_22/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.multi_head_attention_3/dense_22/Tensordot/ProdProd;multi_head_attention_3/dense_22/Tensordot/GatherV2:output:08multi_head_attention_3/dense_22/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_attention_3/dense_22/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
0multi_head_attention_3/dense_22/Tensordot/Prod_1Prod=multi_head_attention_3/dense_22/Tensordot/GatherV2_1:output:0:multi_head_attention_3/dense_22/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5multi_head_attention_3/dense_22/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0multi_head_attention_3/dense_22/Tensordot/concatConcatV27multi_head_attention_3/dense_22/Tensordot/free:output:07multi_head_attention_3/dense_22/Tensordot/axes:output:0>multi_head_attention_3/dense_22/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
/multi_head_attention_3/dense_22/Tensordot/stackPack7multi_head_attention_3/dense_22/Tensordot/Prod:output:09multi_head_attention_3/dense_22/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
3multi_head_attention_3/dense_22/Tensordot/transpose	Transposeinputs9multi_head_attention_3/dense_22/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
1multi_head_attention_3/dense_22/Tensordot/ReshapeReshape7multi_head_attention_3/dense_22/Tensordot/transpose:y:08multi_head_attention_3/dense_22/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
0multi_head_attention_3/dense_22/Tensordot/MatMulMatMul:multi_head_attention_3/dense_22/Tensordot/Reshape:output:0@multi_head_attention_3/dense_22/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????{
1multi_head_attention_3/dense_22/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:y
7multi_head_attention_3/dense_22/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_attention_3/dense_22/Tensordot/concat_1ConcatV2;multi_head_attention_3/dense_22/Tensordot/GatherV2:output:0:multi_head_attention_3/dense_22/Tensordot/Const_2:output:0@multi_head_attention_3/dense_22/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
)multi_head_attention_3/dense_22/TensordotReshape:multi_head_attention_3/dense_22/Tensordot/MatMul:product:0;multi_head_attention_3/dense_22/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
6multi_head_attention_3/dense_22/BiasAdd/ReadVariableOpReadVariableOp?multi_head_attention_3_dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'multi_head_attention_3/dense_22/BiasAddBiasAdd2multi_head_attention_3/dense_22/Tensordot:output:0>multi_head_attention_3/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'q
&multi_head_attention_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????h
&multi_head_attention_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :h
&multi_head_attention_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
$multi_head_attention_3/Reshape/shapePack-multi_head_attention_3/strided_slice:output:0/multi_head_attention_3/Reshape/shape/1:output:0/multi_head_attention_3/Reshape/shape/2:output:0/multi_head_attention_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
multi_head_attention_3/ReshapeReshape0multi_head_attention_3/dense_20/BiasAdd:output:0-multi_head_attention_3/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????~
%multi_head_attention_3/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
 multi_head_attention_3/transpose	Transpose'multi_head_attention_3/Reshape:output:0.multi_head_attention_3/transpose/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????s
(multi_head_attention_3/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????j
(multi_head_attention_3/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :j
(multi_head_attention_3/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
&multi_head_attention_3/Reshape_1/shapePack-multi_head_attention_3/strided_slice:output:01multi_head_attention_3/Reshape_1/shape/1:output:01multi_head_attention_3/Reshape_1/shape/2:output:01multi_head_attention_3/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:?
 multi_head_attention_3/Reshape_1Reshape0multi_head_attention_3/dense_21/BiasAdd:output:0/multi_head_attention_3/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"???????????????????
'multi_head_attention_3/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
"multi_head_attention_3/transpose_1	Transpose)multi_head_attention_3/Reshape_1:output:00multi_head_attention_3/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????s
(multi_head_attention_3/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????j
(multi_head_attention_3/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :j
(multi_head_attention_3/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
&multi_head_attention_3/Reshape_2/shapePack-multi_head_attention_3/strided_slice:output:01multi_head_attention_3/Reshape_2/shape/1:output:01multi_head_attention_3/Reshape_2/shape/2:output:01multi_head_attention_3/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:?
 multi_head_attention_3/Reshape_2Reshape0multi_head_attention_3/dense_22/BiasAdd:output:0/multi_head_attention_3/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"???????????????????
'multi_head_attention_3/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
"multi_head_attention_3/transpose_2	Transpose)multi_head_attention_3/Reshape_2:output:00multi_head_attention_3/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"???????????????????
multi_head_attention_3/MatMulBatchMatMulV2$multi_head_attention_3/transpose:y:0&multi_head_attention_3/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(t
multi_head_attention_3/Shape_1Shape&multi_head_attention_3/transpose_1:y:0*
T0*
_output_shapes
:
,multi_head_attention_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????x
.multi_head_attention_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.multi_head_attention_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&multi_head_attention_3/strided_slice_1StridedSlice'multi_head_attention_3/Shape_1:output:05multi_head_attention_3/strided_slice_1/stack:output:07multi_head_attention_3/strided_slice_1/stack_1:output:07multi_head_attention_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
multi_head_attention_3/CastCast/multi_head_attention_3/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: e
multi_head_attention_3/SqrtSqrtmulti_head_attention_3/Cast:y:0*
T0*
_output_shapes
: ?
multi_head_attention_3/truedivRealDiv&multi_head_attention_3/MatMul:output:0multi_head_attention_3/Sqrt:y:0*
T0*A
_output_shapes/
-:+????????????????????????????
multi_head_attention_3/SoftmaxSoftmax"multi_head_attention_3/truediv:z:0*
T0*A
_output_shapes/
-:+????????????????????????????
multi_head_attention_3/MatMul_1BatchMatMulV2(multi_head_attention_3/Softmax:softmax:0&multi_head_attention_3/transpose_2:y:0*
T0*8
_output_shapes&
$:"???????????????????
'multi_head_attention_3/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
"multi_head_attention_3/transpose_3	Transpose(multi_head_attention_3/MatMul_1:output:00multi_head_attention_3/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????s
(multi_head_attention_3/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????j
(multi_head_attention_3/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
&multi_head_attention_3/Reshape_3/shapePack-multi_head_attention_3/strided_slice:output:01multi_head_attention_3/Reshape_3/shape/1:output:01multi_head_attention_3/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:?
 multi_head_attention_3/Reshape_3Reshape&multi_head_attention_3/transpose_3:y:0/multi_head_attention_3/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :???????????????????
8multi_head_attention_3/dense_23/Tensordot/ReadVariableOpReadVariableOpAmulti_head_attention_3_dense_23_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0x
.multi_head_attention_3/dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
.multi_head_attention_3/dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
/multi_head_attention_3/dense_23/Tensordot/ShapeShape)multi_head_attention_3/Reshape_3:output:0*
T0*
_output_shapes
:y
7multi_head_attention_3/dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_attention_3/dense_23/Tensordot/GatherV2GatherV28multi_head_attention_3/dense_23/Tensordot/Shape:output:07multi_head_attention_3/dense_23/Tensordot/free:output:0@multi_head_attention_3/dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9multi_head_attention_3/dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4multi_head_attention_3/dense_23/Tensordot/GatherV2_1GatherV28multi_head_attention_3/dense_23/Tensordot/Shape:output:07multi_head_attention_3/dense_23/Tensordot/axes:output:0Bmulti_head_attention_3/dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/multi_head_attention_3/dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.multi_head_attention_3/dense_23/Tensordot/ProdProd;multi_head_attention_3/dense_23/Tensordot/GatherV2:output:08multi_head_attention_3/dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_attention_3/dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
0multi_head_attention_3/dense_23/Tensordot/Prod_1Prod=multi_head_attention_3/dense_23/Tensordot/GatherV2_1:output:0:multi_head_attention_3/dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5multi_head_attention_3/dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0multi_head_attention_3/dense_23/Tensordot/concatConcatV27multi_head_attention_3/dense_23/Tensordot/free:output:07multi_head_attention_3/dense_23/Tensordot/axes:output:0>multi_head_attention_3/dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
/multi_head_attention_3/dense_23/Tensordot/stackPack7multi_head_attention_3/dense_23/Tensordot/Prod:output:09multi_head_attention_3/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
3multi_head_attention_3/dense_23/Tensordot/transpose	Transpose)multi_head_attention_3/Reshape_3:output:09multi_head_attention_3/dense_23/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :???????????????????
1multi_head_attention_3/dense_23/Tensordot/ReshapeReshape7multi_head_attention_3/dense_23/Tensordot/transpose:y:08multi_head_attention_3/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
0multi_head_attention_3/dense_23/Tensordot/MatMulMatMul:multi_head_attention_3/dense_23/Tensordot/Reshape:output:0@multi_head_attention_3/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????{
1multi_head_attention_3/dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:y
7multi_head_attention_3/dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2multi_head_attention_3/dense_23/Tensordot/concat_1ConcatV2;multi_head_attention_3/dense_23/Tensordot/GatherV2:output:0:multi_head_attention_3/dense_23/Tensordot/Const_2:output:0@multi_head_attention_3/dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
)multi_head_attention_3/dense_23/TensordotReshape:multi_head_attention_3/dense_23/Tensordot/MatMul:product:0;multi_head_attention_3/dense_23/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :???????????????????
6multi_head_attention_3/dense_23/BiasAdd/ReadVariableOpReadVariableOp?multi_head_attention_3_dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'multi_head_attention_3/dense_23/BiasAddBiasAdd2multi_head_attention_3/dense_23/Tensordot:output:0>multi_head_attention_3/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????\
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_8/dropout/MulMul0multi_head_attention_3/dense_23/BiasAdd:output:0 dropout_8/dropout/Const:output:0*
T0*4
_output_shapes"
 :??????????????????w
dropout_8/dropout/ShapeShape0multi_head_attention_3/dense_23/BiasAdd:output:0*
T0*
_output_shapes
:?
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype0e
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :???????????????????
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :???????????????????
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*4
_output_shapes"
 :??????????????????h
addAddV2inputsdropout_8/dropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????'~
4layer_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
"layer_normalization_6/moments/meanMeanadd:z:0=layer_normalization_6/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????'*
	keep_dims(?
*layer_normalization_6/moments/StopGradientStopGradient+layer_normalization_6/moments/mean:output:0*
T0*,
_output_shapes
:??????????'?
/layer_normalization_6/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_6/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????'?
8layer_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_6/moments/varianceMean3layer_normalization_6/moments/SquaredDifference:z:0Alayer_normalization_6/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????'*
	keep_dims(j
%layer_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?5?
#layer_normalization_6/batchnorm/addAddV2/layer_normalization_6/moments/variance:output:0.layer_normalization_6/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????'?
%layer_normalization_6/batchnorm/RsqrtRsqrt'layer_normalization_6/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????'?
2layer_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
#layer_normalization_6/batchnorm/mulMul)layer_normalization_6/batchnorm/Rsqrt:y:0:layer_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
%layer_normalization_6/batchnorm/mul_1Muladd:z:0'layer_normalization_6/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????'?
%layer_normalization_6/batchnorm/mul_2Mul+layer_normalization_6/moments/mean:output:0'layer_normalization_6/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????'?
.layer_normalization_6/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
#layer_normalization_6/batchnorm/subSub6layer_normalization_6/batchnorm/ReadVariableOp:value:0)layer_normalization_6/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????'?
%layer_normalization_6/batchnorm/add_1AddV2)layer_normalization_6/batchnorm/mul_1:z:0'layer_normalization_6/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????'?
.sequential_3/dense_24/Tensordot/ReadVariableOpReadVariableOp7sequential_3_dense_24_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0n
$sequential_3/dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:u
$sequential_3/dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ~
%sequential_3/dense_24/Tensordot/ShapeShape)layer_normalization_6/batchnorm/add_1:z:0*
T0*
_output_shapes
:o
-sequential_3/dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(sequential_3/dense_24/Tensordot/GatherV2GatherV2.sequential_3/dense_24/Tensordot/Shape:output:0-sequential_3/dense_24/Tensordot/free:output:06sequential_3/dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:q
/sequential_3/dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
*sequential_3/dense_24/Tensordot/GatherV2_1GatherV2.sequential_3/dense_24/Tensordot/Shape:output:0-sequential_3/dense_24/Tensordot/axes:output:08sequential_3/dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
%sequential_3/dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
$sequential_3/dense_24/Tensordot/ProdProd1sequential_3/dense_24/Tensordot/GatherV2:output:0.sequential_3/dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'sequential_3/dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
&sequential_3/dense_24/Tensordot/Prod_1Prod3sequential_3/dense_24/Tensordot/GatherV2_1:output:00sequential_3/dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+sequential_3/dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
&sequential_3/dense_24/Tensordot/concatConcatV2-sequential_3/dense_24/Tensordot/free:output:0-sequential_3/dense_24/Tensordot/axes:output:04sequential_3/dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
%sequential_3/dense_24/Tensordot/stackPack-sequential_3/dense_24/Tensordot/Prod:output:0/sequential_3/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
)sequential_3/dense_24/Tensordot/transpose	Transpose)layer_normalization_6/batchnorm/add_1:z:0/sequential_3/dense_24/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
'sequential_3/dense_24/Tensordot/ReshapeReshape-sequential_3/dense_24/Tensordot/transpose:y:0.sequential_3/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
&sequential_3/dense_24/Tensordot/MatMulMatMul0sequential_3/dense_24/Tensordot/Reshape:output:06sequential_3/dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????q
'sequential_3/dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:o
-sequential_3/dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(sequential_3/dense_24/Tensordot/concat_1ConcatV21sequential_3/dense_24/Tensordot/GatherV2:output:00sequential_3/dense_24/Tensordot/Const_2:output:06sequential_3/dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
sequential_3/dense_24/TensordotReshape0sequential_3/dense_24/Tensordot/MatMul:product:01sequential_3/dense_24/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
,sequential_3/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_3/dense_24/BiasAddBiasAdd(sequential_3/dense_24/Tensordot:output:04sequential_3/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
sequential_3/dense_24/ReluRelu&sequential_3/dense_24/BiasAdd:output:0*
T0*,
_output_shapes
:??????????'?
.sequential_3/dense_25/Tensordot/ReadVariableOpReadVariableOp7sequential_3_dense_25_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0n
$sequential_3/dense_25/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:u
$sequential_3/dense_25/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
%sequential_3/dense_25/Tensordot/ShapeShape(sequential_3/dense_24/Relu:activations:0*
T0*
_output_shapes
:o
-sequential_3/dense_25/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(sequential_3/dense_25/Tensordot/GatherV2GatherV2.sequential_3/dense_25/Tensordot/Shape:output:0-sequential_3/dense_25/Tensordot/free:output:06sequential_3/dense_25/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:q
/sequential_3/dense_25/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
*sequential_3/dense_25/Tensordot/GatherV2_1GatherV2.sequential_3/dense_25/Tensordot/Shape:output:0-sequential_3/dense_25/Tensordot/axes:output:08sequential_3/dense_25/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
%sequential_3/dense_25/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
$sequential_3/dense_25/Tensordot/ProdProd1sequential_3/dense_25/Tensordot/GatherV2:output:0.sequential_3/dense_25/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'sequential_3/dense_25/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
&sequential_3/dense_25/Tensordot/Prod_1Prod3sequential_3/dense_25/Tensordot/GatherV2_1:output:00sequential_3/dense_25/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+sequential_3/dense_25/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
&sequential_3/dense_25/Tensordot/concatConcatV2-sequential_3/dense_25/Tensordot/free:output:0-sequential_3/dense_25/Tensordot/axes:output:04sequential_3/dense_25/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
%sequential_3/dense_25/Tensordot/stackPack-sequential_3/dense_25/Tensordot/Prod:output:0/sequential_3/dense_25/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
)sequential_3/dense_25/Tensordot/transpose	Transpose(sequential_3/dense_24/Relu:activations:0/sequential_3/dense_25/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
'sequential_3/dense_25/Tensordot/ReshapeReshape-sequential_3/dense_25/Tensordot/transpose:y:0.sequential_3/dense_25/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
&sequential_3/dense_25/Tensordot/MatMulMatMul0sequential_3/dense_25/Tensordot/Reshape:output:06sequential_3/dense_25/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????q
'sequential_3/dense_25/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:o
-sequential_3/dense_25/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(sequential_3/dense_25/Tensordot/concat_1ConcatV21sequential_3/dense_25/Tensordot/GatherV2:output:00sequential_3/dense_25/Tensordot/Const_2:output:06sequential_3/dense_25/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
sequential_3/dense_25/TensordotReshape0sequential_3/dense_25/Tensordot/MatMul:product:01sequential_3/dense_25/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
,sequential_3/dense_25/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_3/dense_25/BiasAddBiasAdd(sequential_3/dense_25/Tensordot:output:04sequential_3/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'\
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_9/dropout/MulMul&sequential_3/dense_25/BiasAdd:output:0 dropout_9/dropout/Const:output:0*
T0*,
_output_shapes
:??????????'m
dropout_9/dropout/ShapeShape&sequential_3/dense_25/BiasAdd:output:0*
T0*
_output_shapes
:?
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????'*
dtype0e
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????'?
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????'?
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????'?
add_1AddV2)layer_normalization_6/batchnorm/add_1:z:0dropout_9/dropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????'~
4layer_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
"layer_normalization_7/moments/meanMean	add_1:z:0=layer_normalization_7/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????'*
	keep_dims(?
*layer_normalization_7/moments/StopGradientStopGradient+layer_normalization_7/moments/mean:output:0*
T0*,
_output_shapes
:??????????'?
/layer_normalization_7/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_7/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????'?
8layer_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_7/moments/varianceMean3layer_normalization_7/moments/SquaredDifference:z:0Alayer_normalization_7/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????'*
	keep_dims(j
%layer_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?5?
#layer_normalization_7/batchnorm/addAddV2/layer_normalization_7/moments/variance:output:0.layer_normalization_7/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????'?
%layer_normalization_7/batchnorm/RsqrtRsqrt'layer_normalization_7/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????'?
2layer_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
#layer_normalization_7/batchnorm/mulMul)layer_normalization_7/batchnorm/Rsqrt:y:0:layer_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'?
%layer_normalization_7/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_7/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????'?
%layer_normalization_7/batchnorm/mul_2Mul+layer_normalization_7/moments/mean:output:0'layer_normalization_7/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????'?
.layer_normalization_7/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
#layer_normalization_7/batchnorm/subSub6layer_normalization_7/batchnorm/ReadVariableOp:value:0)layer_normalization_7/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????'?
%layer_normalization_7/batchnorm/add_1AddV2)layer_normalization_7/batchnorm/mul_1:z:0'layer_normalization_7/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????'}
IdentityIdentity)layer_normalization_7/batchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:??????????'?
NoOpNoOp/^layer_normalization_6/batchnorm/ReadVariableOp3^layer_normalization_6/batchnorm/mul/ReadVariableOp/^layer_normalization_7/batchnorm/ReadVariableOp3^layer_normalization_7/batchnorm/mul/ReadVariableOp7^multi_head_attention_3/dense_20/BiasAdd/ReadVariableOp9^multi_head_attention_3/dense_20/Tensordot/ReadVariableOp7^multi_head_attention_3/dense_21/BiasAdd/ReadVariableOp9^multi_head_attention_3/dense_21/Tensordot/ReadVariableOp7^multi_head_attention_3/dense_22/BiasAdd/ReadVariableOp9^multi_head_attention_3/dense_22/Tensordot/ReadVariableOp7^multi_head_attention_3/dense_23/BiasAdd/ReadVariableOp9^multi_head_attention_3/dense_23/Tensordot/ReadVariableOp-^sequential_3/dense_24/BiasAdd/ReadVariableOp/^sequential_3/dense_24/Tensordot/ReadVariableOp-^sequential_3/dense_25/BiasAdd/ReadVariableOp/^sequential_3/dense_25/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:??????????': : : : : : : : : : : : : : : : 2`
.layer_normalization_6/batchnorm/ReadVariableOp.layer_normalization_6/batchnorm/ReadVariableOp2h
2layer_normalization_6/batchnorm/mul/ReadVariableOp2layer_normalization_6/batchnorm/mul/ReadVariableOp2`
.layer_normalization_7/batchnorm/ReadVariableOp.layer_normalization_7/batchnorm/ReadVariableOp2h
2layer_normalization_7/batchnorm/mul/ReadVariableOp2layer_normalization_7/batchnorm/mul/ReadVariableOp2p
6multi_head_attention_3/dense_20/BiasAdd/ReadVariableOp6multi_head_attention_3/dense_20/BiasAdd/ReadVariableOp2t
8multi_head_attention_3/dense_20/Tensordot/ReadVariableOp8multi_head_attention_3/dense_20/Tensordot/ReadVariableOp2p
6multi_head_attention_3/dense_21/BiasAdd/ReadVariableOp6multi_head_attention_3/dense_21/BiasAdd/ReadVariableOp2t
8multi_head_attention_3/dense_21/Tensordot/ReadVariableOp8multi_head_attention_3/dense_21/Tensordot/ReadVariableOp2p
6multi_head_attention_3/dense_22/BiasAdd/ReadVariableOp6multi_head_attention_3/dense_22/BiasAdd/ReadVariableOp2t
8multi_head_attention_3/dense_22/Tensordot/ReadVariableOp8multi_head_attention_3/dense_22/Tensordot/ReadVariableOp2p
6multi_head_attention_3/dense_23/BiasAdd/ReadVariableOp6multi_head_attention_3/dense_23/BiasAdd/ReadVariableOp2t
8multi_head_attention_3/dense_23/Tensordot/ReadVariableOp8multi_head_attention_3/dense_23/Tensordot/ReadVariableOp2\
,sequential_3/dense_24/BiasAdd/ReadVariableOp,sequential_3/dense_24/BiasAdd/ReadVariableOp2`
.sequential_3/dense_24/Tensordot/ReadVariableOp.sequential_3/dense_24/Tensordot/ReadVariableOp2\
,sequential_3/dense_25/BiasAdd/ReadVariableOp,sequential_3/dense_25/BiasAdd/ReadVariableOp2`
.sequential_3/dense_25/Tensordot/ReadVariableOp.sequential_3/dense_25/Tensordot/ReadVariableOp:T P
,
_output_shapes
:??????????'
 
_user_specified_nameinputs
?
?
Z__inference_token_and_position_embedding_14_layer_call_and_return_conditional_losses_17012
x6
#embedding_33_embedding_lookup_16999:	?N6
#embedding_32_embedding_lookup_17005:	?
identity??embedding_32/embedding_lookup?embedding_33/embedding_lookuph
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
strided_sliceStridedSlicexstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:??????????'*

begin_mask*
end_maskj
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
strided_slice_1StridedSlicexstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*,
_output_shapes
:??????????'*

begin_mask*
end_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  u
ReshapeReshapestrided_slice:output:0Reshape/shape:output:0*
T0*(
_output_shapes
:??????????'`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  {
	Reshape_1Reshapestrided_slice_1:output:0Reshape_1/shape:output:0*
T0*(
_output_shapes
:??????????'o
embedding_33/CastCastReshape_1:output:0*

DstT0*

SrcT0*(
_output_shapes
:??????????'?
embedding_33/embedding_lookupResourceGather#embedding_33_embedding_lookup_16999embedding_33/Cast:y:0*
Tindices0*6
_class,
*(loc:@embedding_33/embedding_lookup/16999*,
_output_shapes
:??????????'*
dtype0?
&embedding_33/embedding_lookup/IdentityIdentity&embedding_33/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_33/embedding_lookup/16999*,
_output_shapes
:??????????'?
(embedding_33/embedding_lookup/Identity_1Identity/embedding_33/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????'m
embedding_32/CastCastReshape:output:0*

DstT0*

SrcT0*(
_output_shapes
:??????????'?
embedding_32/embedding_lookupResourceGather#embedding_32_embedding_lookup_17005embedding_32/Cast:y:0*
Tindices0*6
_class,
*(loc:@embedding_32/embedding_lookup/17005*,
_output_shapes
:??????????'*
dtype0?
&embedding_32/embedding_lookup/IdentityIdentity&embedding_32/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_32/embedding_lookup/17005*,
_output_shapes
:??????????'?
(embedding_32/embedding_lookup/Identity_1Identity/embedding_32/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????'?
addAddV21embedding_32/embedding_lookup/Identity_1:output:01embedding_33/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????'[
IdentityIdentityadd:z:0^NoOp*
T0*,
_output_shapes
:??????????'?
NoOpNoOp^embedding_32/embedding_lookup^embedding_33/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????': : 2>
embedding_32/embedding_lookupembedding_32/embedding_lookup2>
embedding_33/embedding_lookupembedding_33/embedding_lookup:O K
,
_output_shapes
:??????????'

_user_specified_namex
?
?
'__inference_model_1_layer_call_fn_18230

inputs
unknown:	?N
	unknown_0:	?
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_17351o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:??????????': : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????'
 
_user_specified_nameinputs
?	
?
C__inference_dense_27_layer_call_and_return_conditional_losses_19608

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_3_layer_call_and_return_conditional_losses_16911

inputs 
dense_24_16900:
dense_24_16902: 
dense_25_16905:
dense_25_16907:
identity?? dense_24/StatefulPartitionedCall? dense_25/StatefulPartitionedCall?
 dense_24/StatefulPartitionedCallStatefulPartitionedCallinputsdense_24_16900dense_24_16902*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_16808?
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_16905dense_25_16907*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_25_layer_call_and_return_conditional_losses_16844}
IdentityIdentity)dense_25/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????'?
NoOpNoOp!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????': : : : 2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall:T P
,
_output_shapes
:??????????'
 
_user_specified_nameinputs
?

?
C__inference_dense_26_layer_call_and_return_conditional_losses_17321

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_dense_25_layer_call_fn_19797

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_25_layer_call_and_return_conditional_losses_16844t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????'`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????': : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????'
 
_user_specified_nameinputs
??
?9
!__inference__traced_restore_20298
file_prefix2
 assignvariableop_dense_26_kernel:.
 assignvariableop_1_dense_26_bias:4
"assignvariableop_2_dense_27_kernel:.
 assignvariableop_3_dense_27_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: ]
Jassignvariableop_9_token_and_position_embedding_14_embedding_32_embeddings:	?^
Kassignvariableop_10_token_and_position_embedding_14_embedding_33_embeddings:	?N`
Nassignvariableop_11_transformer_block_3_multi_head_attention_3_dense_20_kernel:Z
Lassignvariableop_12_transformer_block_3_multi_head_attention_3_dense_20_bias:`
Nassignvariableop_13_transformer_block_3_multi_head_attention_3_dense_21_kernel:Z
Lassignvariableop_14_transformer_block_3_multi_head_attention_3_dense_21_bias:`
Nassignvariableop_15_transformer_block_3_multi_head_attention_3_dense_22_kernel:Z
Lassignvariableop_16_transformer_block_3_multi_head_attention_3_dense_22_bias:`
Nassignvariableop_17_transformer_block_3_multi_head_attention_3_dense_23_kernel:Z
Lassignvariableop_18_transformer_block_3_multi_head_attention_3_dense_23_bias:5
#assignvariableop_19_dense_24_kernel:/
!assignvariableop_20_dense_24_bias:5
#assignvariableop_21_dense_25_kernel:/
!assignvariableop_22_dense_25_bias:Q
Cassignvariableop_23_transformer_block_3_layer_normalization_6_gamma:P
Bassignvariableop_24_transformer_block_3_layer_normalization_6_beta:Q
Cassignvariableop_25_transformer_block_3_layer_normalization_7_gamma:P
Bassignvariableop_26_transformer_block_3_layer_normalization_7_beta:#
assignvariableop_27_total: #
assignvariableop_28_count: <
*assignvariableop_29_adam_dense_26_kernel_m:6
(assignvariableop_30_adam_dense_26_bias_m:<
*assignvariableop_31_adam_dense_27_kernel_m:6
(assignvariableop_32_adam_dense_27_bias_m:e
Rassignvariableop_33_adam_token_and_position_embedding_14_embedding_32_embeddings_m:	?e
Rassignvariableop_34_adam_token_and_position_embedding_14_embedding_33_embeddings_m:	?Ng
Uassignvariableop_35_adam_transformer_block_3_multi_head_attention_3_dense_20_kernel_m:a
Sassignvariableop_36_adam_transformer_block_3_multi_head_attention_3_dense_20_bias_m:g
Uassignvariableop_37_adam_transformer_block_3_multi_head_attention_3_dense_21_kernel_m:a
Sassignvariableop_38_adam_transformer_block_3_multi_head_attention_3_dense_21_bias_m:g
Uassignvariableop_39_adam_transformer_block_3_multi_head_attention_3_dense_22_kernel_m:a
Sassignvariableop_40_adam_transformer_block_3_multi_head_attention_3_dense_22_bias_m:g
Uassignvariableop_41_adam_transformer_block_3_multi_head_attention_3_dense_23_kernel_m:a
Sassignvariableop_42_adam_transformer_block_3_multi_head_attention_3_dense_23_bias_m:<
*assignvariableop_43_adam_dense_24_kernel_m:6
(assignvariableop_44_adam_dense_24_bias_m:<
*assignvariableop_45_adam_dense_25_kernel_m:6
(assignvariableop_46_adam_dense_25_bias_m:X
Jassignvariableop_47_adam_transformer_block_3_layer_normalization_6_gamma_m:W
Iassignvariableop_48_adam_transformer_block_3_layer_normalization_6_beta_m:X
Jassignvariableop_49_adam_transformer_block_3_layer_normalization_7_gamma_m:W
Iassignvariableop_50_adam_transformer_block_3_layer_normalization_7_beta_m:<
*assignvariableop_51_adam_dense_26_kernel_v:6
(assignvariableop_52_adam_dense_26_bias_v:<
*assignvariableop_53_adam_dense_27_kernel_v:6
(assignvariableop_54_adam_dense_27_bias_v:e
Rassignvariableop_55_adam_token_and_position_embedding_14_embedding_32_embeddings_v:	?e
Rassignvariableop_56_adam_token_and_position_embedding_14_embedding_33_embeddings_v:	?Ng
Uassignvariableop_57_adam_transformer_block_3_multi_head_attention_3_dense_20_kernel_v:a
Sassignvariableop_58_adam_transformer_block_3_multi_head_attention_3_dense_20_bias_v:g
Uassignvariableop_59_adam_transformer_block_3_multi_head_attention_3_dense_21_kernel_v:a
Sassignvariableop_60_adam_transformer_block_3_multi_head_attention_3_dense_21_bias_v:g
Uassignvariableop_61_adam_transformer_block_3_multi_head_attention_3_dense_22_kernel_v:a
Sassignvariableop_62_adam_transformer_block_3_multi_head_attention_3_dense_22_bias_v:g
Uassignvariableop_63_adam_transformer_block_3_multi_head_attention_3_dense_23_kernel_v:a
Sassignvariableop_64_adam_transformer_block_3_multi_head_attention_3_dense_23_bias_v:<
*assignvariableop_65_adam_dense_24_kernel_v:6
(assignvariableop_66_adam_dense_24_bias_v:<
*assignvariableop_67_adam_dense_25_kernel_v:6
(assignvariableop_68_adam_dense_25_bias_v:X
Jassignvariableop_69_adam_transformer_block_3_layer_normalization_6_gamma_v:W
Iassignvariableop_70_adam_transformer_block_3_layer_normalization_6_beta_v:X
Jassignvariableop_71_adam_transformer_block_3_layer_normalization_7_gamma_v:W
Iassignvariableop_72_adam_transformer_block_3_layer_normalization_7_beta_v:
identity_74??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_8?AssignVariableOp_9?#
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*?"
value?"B?"JB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*?
value?B?JB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*X
dtypesN
L2J	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp assignvariableop_dense_26_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_26_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_27_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_27_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpJassignvariableop_9_token_and_position_embedding_14_embedding_32_embeddingsIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpKassignvariableop_10_token_and_position_embedding_14_embedding_33_embeddingsIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpNassignvariableop_11_transformer_block_3_multi_head_attention_3_dense_20_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpLassignvariableop_12_transformer_block_3_multi_head_attention_3_dense_20_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpNassignvariableop_13_transformer_block_3_multi_head_attention_3_dense_21_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpLassignvariableop_14_transformer_block_3_multi_head_attention_3_dense_21_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpNassignvariableop_15_transformer_block_3_multi_head_attention_3_dense_22_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpLassignvariableop_16_transformer_block_3_multi_head_attention_3_dense_22_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpNassignvariableop_17_transformer_block_3_multi_head_attention_3_dense_23_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpLassignvariableop_18_transformer_block_3_multi_head_attention_3_dense_23_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp#assignvariableop_19_dense_24_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp!assignvariableop_20_dense_24_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp#assignvariableop_21_dense_25_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp!assignvariableop_22_dense_25_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOpCassignvariableop_23_transformer_block_3_layer_normalization_6_gammaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOpBassignvariableop_24_transformer_block_3_layer_normalization_6_betaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOpCassignvariableop_25_transformer_block_3_layer_normalization_7_gammaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpBassignvariableop_26_transformer_block_3_layer_normalization_7_betaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_26_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_26_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_27_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_27_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOpRassignvariableop_33_adam_token_and_position_embedding_14_embedding_32_embeddings_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOpRassignvariableop_34_adam_token_and_position_embedding_14_embedding_33_embeddings_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOpUassignvariableop_35_adam_transformer_block_3_multi_head_attention_3_dense_20_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOpSassignvariableop_36_adam_transformer_block_3_multi_head_attention_3_dense_20_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOpUassignvariableop_37_adam_transformer_block_3_multi_head_attention_3_dense_21_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOpSassignvariableop_38_adam_transformer_block_3_multi_head_attention_3_dense_21_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOpUassignvariableop_39_adam_transformer_block_3_multi_head_attention_3_dense_22_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOpSassignvariableop_40_adam_transformer_block_3_multi_head_attention_3_dense_22_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOpUassignvariableop_41_adam_transformer_block_3_multi_head_attention_3_dense_23_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOpSassignvariableop_42_adam_transformer_block_3_multi_head_attention_3_dense_23_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_dense_24_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_dense_24_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_dense_25_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_dense_25_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOpJassignvariableop_47_adam_transformer_block_3_layer_normalization_6_gamma_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOpIassignvariableop_48_adam_transformer_block_3_layer_normalization_6_beta_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOpJassignvariableop_49_adam_transformer_block_3_layer_normalization_7_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOpIassignvariableop_50_adam_transformer_block_3_layer_normalization_7_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_26_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_26_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_dense_27_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_dense_27_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOpRassignvariableop_55_adam_token_and_position_embedding_14_embedding_32_embeddings_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOpRassignvariableop_56_adam_token_and_position_embedding_14_embedding_33_embeddings_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOpUassignvariableop_57_adam_transformer_block_3_multi_head_attention_3_dense_20_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOpSassignvariableop_58_adam_transformer_block_3_multi_head_attention_3_dense_20_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOpUassignvariableop_59_adam_transformer_block_3_multi_head_attention_3_dense_21_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOpSassignvariableop_60_adam_transformer_block_3_multi_head_attention_3_dense_21_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOpUassignvariableop_61_adam_transformer_block_3_multi_head_attention_3_dense_22_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOpSassignvariableop_62_adam_transformer_block_3_multi_head_attention_3_dense_22_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOpUassignvariableop_63_adam_transformer_block_3_multi_head_attention_3_dense_23_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOpSassignvariableop_64_adam_transformer_block_3_multi_head_attention_3_dense_23_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_dense_24_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_dense_24_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOp*assignvariableop_67_adam_dense_25_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOp(assignvariableop_68_adam_dense_25_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOpJassignvariableop_69_adam_transformer_block_3_layer_normalization_6_gamma_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOpIassignvariableop_70_adam_transformer_block_3_layer_normalization_6_beta_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOpJassignvariableop_71_adam_transformer_block_3_layer_normalization_7_gamma_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_72AssignVariableOpIassignvariableop_72_adam_transformer_block_3_layer_normalization_7_beta_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_73Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_74IdentityIdentity_73:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_74Identity_74:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
Z__inference_token_and_position_embedding_14_layer_call_and_return_conditional_losses_18917
x6
#embedding_33_embedding_lookup_18904:	?N6
#embedding_32_embedding_lookup_18910:	?
identity??embedding_32/embedding_lookup?embedding_33/embedding_lookuph
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
strided_sliceStridedSlicexstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:??????????'*

begin_mask*
end_maskj
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
strided_slice_1StridedSlicexstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*,
_output_shapes
:??????????'*

begin_mask*
end_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  u
ReshapeReshapestrided_slice:output:0Reshape/shape:output:0*
T0*(
_output_shapes
:??????????'`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  {
	Reshape_1Reshapestrided_slice_1:output:0Reshape_1/shape:output:0*
T0*(
_output_shapes
:??????????'o
embedding_33/CastCastReshape_1:output:0*

DstT0*

SrcT0*(
_output_shapes
:??????????'?
embedding_33/embedding_lookupResourceGather#embedding_33_embedding_lookup_18904embedding_33/Cast:y:0*
Tindices0*6
_class,
*(loc:@embedding_33/embedding_lookup/18904*,
_output_shapes
:??????????'*
dtype0?
&embedding_33/embedding_lookup/IdentityIdentity&embedding_33/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_33/embedding_lookup/18904*,
_output_shapes
:??????????'?
(embedding_33/embedding_lookup/Identity_1Identity/embedding_33/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????'m
embedding_32/CastCastReshape:output:0*

DstT0*

SrcT0*(
_output_shapes
:??????????'?
embedding_32/embedding_lookupResourceGather#embedding_32_embedding_lookup_18910embedding_32/Cast:y:0*
Tindices0*6
_class,
*(loc:@embedding_32/embedding_lookup/18910*,
_output_shapes
:??????????'*
dtype0?
&embedding_32/embedding_lookup/IdentityIdentity&embedding_32/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_32/embedding_lookup/18910*,
_output_shapes
:??????????'?
(embedding_32/embedding_lookup/Identity_1Identity/embedding_32/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????'?
addAddV21embedding_32/embedding_lookup/Identity_1:output:01embedding_33/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????'[
IdentityIdentityadd:z:0^NoOp*
T0*,
_output_shapes
:??????????'?
NoOpNoOp^embedding_32/embedding_lookup^embedding_33/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????': : 2>
embedding_32/embedding_lookupembedding_32/embedding_lookup2>
embedding_33/embedding_lookupembedding_33/embedding_lookup:O K
,
_output_shapes
:??????????'

_user_specified_namex
?*
?	
B__inference_model_1_layer_call_and_return_conditional_losses_18069
input_178
%token_and_position_embedding_14_18017:	?N8
%token_and_position_embedding_14_18019:	?+
transformer_block_3_18022:'
transformer_block_3_18024:+
transformer_block_3_18026:'
transformer_block_3_18028:+
transformer_block_3_18030:'
transformer_block_3_18032:+
transformer_block_3_18034:'
transformer_block_3_18036:'
transformer_block_3_18038:'
transformer_block_3_18040:+
transformer_block_3_18042:'
transformer_block_3_18044:+
transformer_block_3_18046:'
transformer_block_3_18048:'
transformer_block_3_18050:'
transformer_block_3_18052: 
dense_26_18057:
dense_26_18059: 
dense_27_18063:
dense_27_18065:
identity?? dense_26/StatefulPartitionedCall? dense_27/StatefulPartitionedCall?7token_and_position_embedding_14/StatefulPartitionedCall?+transformer_block_3/StatefulPartitionedCall?
7token_and_position_embedding_14/StatefulPartitionedCallStatefulPartitionedCallinput_17%token_and_position_embedding_14_18017%token_and_position_embedding_14_18019*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *c
f^R\
Z__inference_token_and_position_embedding_14_layer_call_and_return_conditional_losses_17012?
+transformer_block_3/StatefulPartitionedCallStatefulPartitionedCall@token_and_position_embedding_14/StatefulPartitionedCall:output:0transformer_block_3_18022transformer_block_3_18024transformer_block_3_18026transformer_block_3_18028transformer_block_3_18030transformer_block_3_18032transformer_block_3_18034transformer_block_3_18036transformer_block_3_18038transformer_block_3_18040transformer_block_3_18042transformer_block_3_18044transformer_block_3_18046transformer_block_3_18048transformer_block_3_18050transformer_block_3_18052*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????'*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_transformer_block_3_layer_call_and_return_conditional_losses_17262?
*global_average_pooling1d_3/PartitionedCallPartitionedCall4transformer_block_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_global_average_pooling1d_3_layer_call_and_return_conditional_losses_17301?
dropout_10/PartitionedCallPartitionedCall3global_average_pooling1d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_10_layer_call_and_return_conditional_losses_17308?
 dense_26/StatefulPartitionedCallStatefulPartitionedCall#dropout_10/PartitionedCall:output:0dense_26_18057dense_26_18059*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_26_layer_call_and_return_conditional_losses_17321?
dropout_11/PartitionedCallPartitionedCall)dense_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_17332?
 dense_27/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0dense_27_18063dense_27_18065*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_27_layer_call_and_return_conditional_losses_17344x
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall8^token_and_position_embedding_14/StatefulPartitionedCall,^transformer_block_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:??????????': : : : : : : : : : : : : : : : : : : : : : 2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2r
7token_and_position_embedding_14/StatefulPartitionedCall7token_and_position_embedding_14/StatefulPartitionedCall2Z
+transformer_block_3/StatefulPartitionedCall+transformer_block_3/StatefulPartitionedCall:V R
,
_output_shapes
:??????????'
"
_user_specified_name
input_17
??
?
G__inference_sequential_3_layer_call_and_return_conditional_losses_19691

inputs<
*dense_24_tensordot_readvariableop_resource:6
(dense_24_biasadd_readvariableop_resource:<
*dense_25_tensordot_readvariableop_resource:6
(dense_25_biasadd_readvariableop_resource:
identity??dense_24/BiasAdd/ReadVariableOp?!dense_24/Tensordot/ReadVariableOp?dense_25/BiasAdd/ReadVariableOp?!dense_25/Tensordot/ReadVariableOp?
!dense_24/Tensordot/ReadVariableOpReadVariableOp*dense_24_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0a
dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       N
dense_24/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:b
 dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_24/Tensordot/GatherV2GatherV2!dense_24/Tensordot/Shape:output:0 dense_24/Tensordot/free:output:0)dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_24/Tensordot/GatherV2_1GatherV2!dense_24/Tensordot/Shape:output:0 dense_24/Tensordot/axes:output:0+dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_24/Tensordot/ProdProd$dense_24/Tensordot/GatherV2:output:0!dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_24/Tensordot/Prod_1Prod&dense_24/Tensordot/GatherV2_1:output:0#dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_24/Tensordot/concatConcatV2 dense_24/Tensordot/free:output:0 dense_24/Tensordot/axes:output:0'dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_24/Tensordot/stackPack dense_24/Tensordot/Prod:output:0"dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_24/Tensordot/transpose	Transposeinputs"dense_24/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
dense_24/Tensordot/ReshapeReshape dense_24/Tensordot/transpose:y:0!dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_24/Tensordot/MatMulMatMul#dense_24/Tensordot/Reshape:output:0)dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_24/Tensordot/concat_1ConcatV2$dense_24/Tensordot/GatherV2:output:0#dense_24/Tensordot/Const_2:output:0)dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_24/TensordotReshape#dense_24/Tensordot/MatMul:product:0$dense_24/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_24/BiasAddBiasAdddense_24/Tensordot:output:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'g
dense_24/ReluReludense_24/BiasAdd:output:0*
T0*,
_output_shapes
:??????????'?
!dense_25/Tensordot/ReadVariableOpReadVariableOp*dense_25_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0a
dense_25/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_25/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       c
dense_25/Tensordot/ShapeShapedense_24/Relu:activations:0*
T0*
_output_shapes
:b
 dense_25/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_25/Tensordot/GatherV2GatherV2!dense_25/Tensordot/Shape:output:0 dense_25/Tensordot/free:output:0)dense_25/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_25/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_25/Tensordot/GatherV2_1GatherV2!dense_25/Tensordot/Shape:output:0 dense_25/Tensordot/axes:output:0+dense_25/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_25/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_25/Tensordot/ProdProd$dense_25/Tensordot/GatherV2:output:0!dense_25/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_25/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_25/Tensordot/Prod_1Prod&dense_25/Tensordot/GatherV2_1:output:0#dense_25/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_25/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_25/Tensordot/concatConcatV2 dense_25/Tensordot/free:output:0 dense_25/Tensordot/axes:output:0'dense_25/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_25/Tensordot/stackPack dense_25/Tensordot/Prod:output:0"dense_25/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_25/Tensordot/transpose	Transposedense_24/Relu:activations:0"dense_25/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
dense_25/Tensordot/ReshapeReshape dense_25/Tensordot/transpose:y:0!dense_25/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_25/Tensordot/MatMulMatMul#dense_25/Tensordot/Reshape:output:0)dense_25/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
dense_25/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_25/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_25/Tensordot/concat_1ConcatV2$dense_25/Tensordot/GatherV2:output:0#dense_25/Tensordot/Const_2:output:0)dense_25/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_25/TensordotReshape#dense_25/Tensordot/MatMul:product:0$dense_25/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_25/BiasAddBiasAdddense_25/Tensordot:output:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'m
IdentityIdentitydense_25/BiasAdd:output:0^NoOp*
T0*,
_output_shapes
:??????????'?
NoOpNoOp ^dense_24/BiasAdd/ReadVariableOp"^dense_24/Tensordot/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp"^dense_25/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????': : : : 2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2F
!dense_24/Tensordot/ReadVariableOp!dense_24/Tensordot/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2F
!dense_25/Tensordot/ReadVariableOp!dense_25/Tensordot/ReadVariableOp:T P
,
_output_shapes
:??????????'
 
_user_specified_nameinputs
?
?
3__inference_transformer_block_3_layer_call_fn_18954

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????'*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_transformer_block_3_layer_call_and_return_conditional_losses_17262t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????'`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:??????????': : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????'
 
_user_specified_nameinputs
?

?
C__inference_dense_26_layer_call_and_return_conditional_losses_19562

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_dense_25_layer_call_and_return_conditional_losses_16844

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'d
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:??????????'z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????': : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????'
 
_user_specified_nameinputs
?
F
*__inference_dropout_10_layer_call_fn_19520

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_10_layer_call_and_return_conditional_losses_17308`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_10_layer_call_and_return_conditional_losses_19530

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_3_layer_call_and_return_conditional_losses_16851

inputs 
dense_24_16809:
dense_24_16811: 
dense_25_16845:
dense_25_16847:
identity?? dense_24/StatefulPartitionedCall? dense_25/StatefulPartitionedCall?
 dense_24/StatefulPartitionedCallStatefulPartitionedCallinputsdense_24_16809dense_24_16811*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_16808?
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_16845dense_25_16847*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_25_layer_call_and_return_conditional_losses_16844}
IdentityIdentity)dense_25/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????'?
NoOpNoOp!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????': : : : 2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall:T P
,
_output_shapes
:??????????'
 
_user_specified_nameinputs
?
c
E__inference_dropout_11_layer_call_and_return_conditional_losses_19577

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
3__inference_transformer_block_3_layer_call_fn_18991

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????'*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_transformer_block_3_layer_call_and_return_conditional_losses_17767t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????'`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:??????????': : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????'
 
_user_specified_nameinputs
?
q
U__inference_global_average_pooling1d_3_layer_call_and_return_conditional_losses_19509

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?	
?
C__inference_dense_27_layer_call_and_return_conditional_losses_17344

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
G__inference_sequential_3_layer_call_and_return_conditional_losses_19748

inputs<
*dense_24_tensordot_readvariableop_resource:6
(dense_24_biasadd_readvariableop_resource:<
*dense_25_tensordot_readvariableop_resource:6
(dense_25_biasadd_readvariableop_resource:
identity??dense_24/BiasAdd/ReadVariableOp?!dense_24/Tensordot/ReadVariableOp?dense_25/BiasAdd/ReadVariableOp?!dense_25/Tensordot/ReadVariableOp?
!dense_24/Tensordot/ReadVariableOpReadVariableOp*dense_24_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0a
dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       N
dense_24/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:b
 dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_24/Tensordot/GatherV2GatherV2!dense_24/Tensordot/Shape:output:0 dense_24/Tensordot/free:output:0)dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_24/Tensordot/GatherV2_1GatherV2!dense_24/Tensordot/Shape:output:0 dense_24/Tensordot/axes:output:0+dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_24/Tensordot/ProdProd$dense_24/Tensordot/GatherV2:output:0!dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_24/Tensordot/Prod_1Prod&dense_24/Tensordot/GatherV2_1:output:0#dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_24/Tensordot/concatConcatV2 dense_24/Tensordot/free:output:0 dense_24/Tensordot/axes:output:0'dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_24/Tensordot/stackPack dense_24/Tensordot/Prod:output:0"dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_24/Tensordot/transpose	Transposeinputs"dense_24/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
dense_24/Tensordot/ReshapeReshape dense_24/Tensordot/transpose:y:0!dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_24/Tensordot/MatMulMatMul#dense_24/Tensordot/Reshape:output:0)dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_24/Tensordot/concat_1ConcatV2$dense_24/Tensordot/GatherV2:output:0#dense_24/Tensordot/Const_2:output:0)dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_24/TensordotReshape#dense_24/Tensordot/MatMul:product:0$dense_24/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_24/BiasAddBiasAdddense_24/Tensordot:output:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'g
dense_24/ReluReludense_24/BiasAdd:output:0*
T0*,
_output_shapes
:??????????'?
!dense_25/Tensordot/ReadVariableOpReadVariableOp*dense_25_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0a
dense_25/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_25/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       c
dense_25/Tensordot/ShapeShapedense_24/Relu:activations:0*
T0*
_output_shapes
:b
 dense_25/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_25/Tensordot/GatherV2GatherV2!dense_25/Tensordot/Shape:output:0 dense_25/Tensordot/free:output:0)dense_25/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_25/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_25/Tensordot/GatherV2_1GatherV2!dense_25/Tensordot/Shape:output:0 dense_25/Tensordot/axes:output:0+dense_25/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_25/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_25/Tensordot/ProdProd$dense_25/Tensordot/GatherV2:output:0!dense_25/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_25/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_25/Tensordot/Prod_1Prod&dense_25/Tensordot/GatherV2_1:output:0#dense_25/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_25/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_25/Tensordot/concatConcatV2 dense_25/Tensordot/free:output:0 dense_25/Tensordot/axes:output:0'dense_25/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_25/Tensordot/stackPack dense_25/Tensordot/Prod:output:0"dense_25/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_25/Tensordot/transpose	Transposedense_24/Relu:activations:0"dense_25/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????'?
dense_25/Tensordot/ReshapeReshape dense_25/Tensordot/transpose:y:0!dense_25/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_25/Tensordot/MatMulMatMul#dense_25/Tensordot/Reshape:output:0)dense_25/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
dense_25/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_25/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_25/Tensordot/concat_1ConcatV2$dense_25/Tensordot/GatherV2:output:0#dense_25/Tensordot/Const_2:output:0)dense_25/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_25/TensordotReshape#dense_25/Tensordot/MatMul:product:0$dense_25/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????'?
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_25/BiasAddBiasAdddense_25/Tensordot:output:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????'m
IdentityIdentitydense_25/BiasAdd:output:0^NoOp*
T0*,
_output_shapes
:??????????'?
NoOpNoOp ^dense_24/BiasAdd/ReadVariableOp"^dense_24/Tensordot/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp"^dense_25/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????': : : : 2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2F
!dense_24/Tensordot/ReadVariableOp!dense_24/Tensordot/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2F
!dense_25/Tensordot/ReadVariableOp!dense_25/Tensordot/ReadVariableOp:T P
,
_output_shapes
:??????????'
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_18181
input_17
unknown:	?N
	unknown_0:	?
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_17unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_16770o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:??????????': : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:??????????'
"
_user_specified_name
input_17
?*
?	
B__inference_model_1_layer_call_and_return_conditional_losses_17351

inputs8
%token_and_position_embedding_14_17013:	?N8
%token_and_position_embedding_14_17015:	?+
transformer_block_3_17263:'
transformer_block_3_17265:+
transformer_block_3_17267:'
transformer_block_3_17269:+
transformer_block_3_17271:'
transformer_block_3_17273:+
transformer_block_3_17275:'
transformer_block_3_17277:'
transformer_block_3_17279:'
transformer_block_3_17281:+
transformer_block_3_17283:'
transformer_block_3_17285:+
transformer_block_3_17287:'
transformer_block_3_17289:'
transformer_block_3_17291:'
transformer_block_3_17293: 
dense_26_17322:
dense_26_17324: 
dense_27_17345:
dense_27_17347:
identity?? dense_26/StatefulPartitionedCall? dense_27/StatefulPartitionedCall?7token_and_position_embedding_14/StatefulPartitionedCall?+transformer_block_3/StatefulPartitionedCall?
7token_and_position_embedding_14/StatefulPartitionedCallStatefulPartitionedCallinputs%token_and_position_embedding_14_17013%token_and_position_embedding_14_17015*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *c
f^R\
Z__inference_token_and_position_embedding_14_layer_call_and_return_conditional_losses_17012?
+transformer_block_3/StatefulPartitionedCallStatefulPartitionedCall@token_and_position_embedding_14/StatefulPartitionedCall:output:0transformer_block_3_17263transformer_block_3_17265transformer_block_3_17267transformer_block_3_17269transformer_block_3_17271transformer_block_3_17273transformer_block_3_17275transformer_block_3_17277transformer_block_3_17279transformer_block_3_17281transformer_block_3_17283transformer_block_3_17285transformer_block_3_17287transformer_block_3_17289transformer_block_3_17291transformer_block_3_17293*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????'*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_transformer_block_3_layer_call_and_return_conditional_losses_17262?
*global_average_pooling1d_3/PartitionedCallPartitionedCall4transformer_block_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_global_average_pooling1d_3_layer_call_and_return_conditional_losses_17301?
dropout_10/PartitionedCallPartitionedCall3global_average_pooling1d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_10_layer_call_and_return_conditional_losses_17308?
 dense_26/StatefulPartitionedCallStatefulPartitionedCall#dropout_10/PartitionedCall:output:0dense_26_17322dense_26_17324*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_26_layer_call_and_return_conditional_losses_17321?
dropout_11/PartitionedCallPartitionedCall)dense_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_17332?
 dense_27/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0dense_27_17345dense_27_17347*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_27_layer_call_and_return_conditional_losses_17344x
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall8^token_and_position_embedding_14/StatefulPartitionedCall,^transformer_block_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:??????????': : : : : : : : : : : : : : : : : : : : : : 2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2r
7token_and_position_embedding_14/StatefulPartitionedCall7token_and_position_embedding_14/StatefulPartitionedCall2Z
+transformer_block_3/StatefulPartitionedCall+transformer_block_3/StatefulPartitionedCall:T P
,
_output_shapes
:??????????'
 
_user_specified_nameinputs
?	
d
E__inference_dropout_11_layer_call_and_return_conditional_losses_19589

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
B
input_176
serving_default_input_17:0??????????'<
dense_270
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
?
	token_emb
pos_emb
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
att
ffn

layernorm1

layernorm2
dropout1
dropout2
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
 trainable_variables
!regularization_losses
"	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#	variables
$trainable_variables
%regularization_losses
&	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
-	variables
.trainable_variables
/regularization_losses
0	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

1kernel
2bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
7iter

8beta_1

9beta_2
	:decay
;learning_rate'm?(m?1m?2m?<m?=m?>m??m?@m?Am?Bm?Cm?Dm?Em?Fm?Gm?Hm?Im?Jm?Km?Lm?Mm?'v?(v?1v?2v?<v?=v?>v??v?@v?Av?Bv?Cv?Dv?Ev?Fv?Gv?Hv?Iv?Jv?Kv?Lv?Mv?"
	optimizer
?
<0
=1
>2
?3
@4
A5
B6
C7
D8
E9
F10
G11
H12
I13
J14
K15
L16
M17
'18
(19
120
221"
trackable_list_wrapper
?
<0
=1
>2
?3
@4
A5
B6
C7
D8
E9
F10
G11
H12
I13
J14
K15
L16
M17
'18
(19
120
221"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics

	variables
trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?
<
embeddings
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
=
embeddings
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
`query_dense
a	key_dense
bvalue_dense
	cdense
d	variables
etrainable_variables
fregularization_losses
g	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
hlayer_with_weights-0
hlayer-0
ilayer_with_weights-1
ilayer-1
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
naxis
	Jgamma
Kbeta
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
saxis
	Lgamma
Mbeta
t	variables
utrainable_variables
vregularization_losses
w	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
|	variables
}trainable_variables
~regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15"
trackable_list_wrapper
?
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
 trainable_variables
!regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
#	variables
$trainable_variables
%regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:2dense_26/kernel
:2dense_26/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
)	variables
*trainable_variables
+regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
-	variables
.trainable_variables
/regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:2dense_27/kernel
:2dense_27/bias
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
3	variables
4trainable_variables
5regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
J:H	?27token_and_position_embedding_14/embedding_32/embeddings
J:H	?N27token_and_position_embedding_14/embedding_33/embeddings
L:J2:transformer_block_3/multi_head_attention_3/dense_20/kernel
F:D28transformer_block_3/multi_head_attention_3/dense_20/bias
L:J2:transformer_block_3/multi_head_attention_3/dense_21/kernel
F:D28transformer_block_3/multi_head_attention_3/dense_21/bias
L:J2:transformer_block_3/multi_head_attention_3/dense_22/kernel
F:D28transformer_block_3/multi_head_attention_3/dense_22/bias
L:J2:transformer_block_3/multi_head_attention_3/dense_23/kernel
F:D28transformer_block_3/multi_head_attention_3/dense_23/bias
!:2dense_24/kernel
:2dense_24/bias
!:2dense_25/kernel
:2dense_25/bias
=:;2/transformer_block_3/layer_normalization_6/gamma
<::2.transformer_block_3/layer_normalization_6/beta
=:;2/transformer_block_3/layer_normalization_7/gamma
<::2.transformer_block_3/layer_normalization_7/beta
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
<0"
trackable_list_wrapper
'
<0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
=0"
trackable_list_wrapper
'
=0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

>kernel
?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

@kernel
Abias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Bkernel
Cbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Dkernel
Ebias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
X
>0
?1
@2
A3
B4
C5
D6
E7"
trackable_list_wrapper
X
>0
?1
@2
A3
B4
C5
D6
E7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
d	variables
etrainable_variables
fregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

Fkernel
Gbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Hkernel
Ibias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
<
F0
G1
H2
I3"
trackable_list_wrapper
<
F0
G1
H2
I3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
j	variables
ktrainable_variables
lregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
o	variables
ptrainable_variables
qregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
t	variables
utrainable_variables
vregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
x	variables
ytrainable_variables
zregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
|	variables
}trainable_variables
~regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
`0
a1
b2
c3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
&:$2Adam/dense_26/kernel/m
 :2Adam/dense_26/bias/m
&:$2Adam/dense_27/kernel/m
 :2Adam/dense_27/bias/m
O:M	?2>Adam/token_and_position_embedding_14/embedding_32/embeddings/m
O:M	?N2>Adam/token_and_position_embedding_14/embedding_33/embeddings/m
Q:O2AAdam/transformer_block_3/multi_head_attention_3/dense_20/kernel/m
K:I2?Adam/transformer_block_3/multi_head_attention_3/dense_20/bias/m
Q:O2AAdam/transformer_block_3/multi_head_attention_3/dense_21/kernel/m
K:I2?Adam/transformer_block_3/multi_head_attention_3/dense_21/bias/m
Q:O2AAdam/transformer_block_3/multi_head_attention_3/dense_22/kernel/m
K:I2?Adam/transformer_block_3/multi_head_attention_3/dense_22/bias/m
Q:O2AAdam/transformer_block_3/multi_head_attention_3/dense_23/kernel/m
K:I2?Adam/transformer_block_3/multi_head_attention_3/dense_23/bias/m
&:$2Adam/dense_24/kernel/m
 :2Adam/dense_24/bias/m
&:$2Adam/dense_25/kernel/m
 :2Adam/dense_25/bias/m
B:@26Adam/transformer_block_3/layer_normalization_6/gamma/m
A:?25Adam/transformer_block_3/layer_normalization_6/beta/m
B:@26Adam/transformer_block_3/layer_normalization_7/gamma/m
A:?25Adam/transformer_block_3/layer_normalization_7/beta/m
&:$2Adam/dense_26/kernel/v
 :2Adam/dense_26/bias/v
&:$2Adam/dense_27/kernel/v
 :2Adam/dense_27/bias/v
O:M	?2>Adam/token_and_position_embedding_14/embedding_32/embeddings/v
O:M	?N2>Adam/token_and_position_embedding_14/embedding_33/embeddings/v
Q:O2AAdam/transformer_block_3/multi_head_attention_3/dense_20/kernel/v
K:I2?Adam/transformer_block_3/multi_head_attention_3/dense_20/bias/v
Q:O2AAdam/transformer_block_3/multi_head_attention_3/dense_21/kernel/v
K:I2?Adam/transformer_block_3/multi_head_attention_3/dense_21/bias/v
Q:O2AAdam/transformer_block_3/multi_head_attention_3/dense_22/kernel/v
K:I2?Adam/transformer_block_3/multi_head_attention_3/dense_22/bias/v
Q:O2AAdam/transformer_block_3/multi_head_attention_3/dense_23/kernel/v
K:I2?Adam/transformer_block_3/multi_head_attention_3/dense_23/bias/v
&:$2Adam/dense_24/kernel/v
 :2Adam/dense_24/bias/v
&:$2Adam/dense_25/kernel/v
 :2Adam/dense_25/bias/v
B:@26Adam/transformer_block_3/layer_normalization_6/gamma/v
A:?25Adam/transformer_block_3/layer_normalization_6/beta/v
B:@26Adam/transformer_block_3/layer_normalization_7/gamma/v
A:?25Adam/transformer_block_3/layer_normalization_7/beta/v
?2?
'__inference_model_1_layer_call_fn_17398
'__inference_model_1_layer_call_fn_18230
'__inference_model_1_layer_call_fn_18279
'__inference_model_1_layer_call_fn_18014?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_model_1_layer_call_and_return_conditional_losses_18565
B__inference_model_1_layer_call_and_return_conditional_losses_18879
B__inference_model_1_layer_call_and_return_conditional_losses_18069
B__inference_model_1_layer_call_and_return_conditional_losses_18124?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
 __inference__wrapped_model_16770input_17"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_token_and_position_embedding_14_layer_call_fn_18888?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Z__inference_token_and_position_embedding_14_layer_call_and_return_conditional_losses_18917?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
3__inference_transformer_block_3_layer_call_fn_18954
3__inference_transformer_block_3_layer_call_fn_18991?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults? 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
N__inference_transformer_block_3_layer_call_and_return_conditional_losses_19235
N__inference_transformer_block_3_layer_call_and_return_conditional_losses_19493?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults? 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
:__inference_global_average_pooling1d_3_layer_call_fn_19498
:__inference_global_average_pooling1d_3_layer_call_fn_19503?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
U__inference_global_average_pooling1d_3_layer_call_and_return_conditional_losses_19509
U__inference_global_average_pooling1d_3_layer_call_and_return_conditional_losses_19515?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dropout_10_layer_call_fn_19520
*__inference_dropout_10_layer_call_fn_19525?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_10_layer_call_and_return_conditional_losses_19530
E__inference_dropout_10_layer_call_and_return_conditional_losses_19542?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dense_26_layer_call_fn_19551?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_26_layer_call_and_return_conditional_losses_19562?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dropout_11_layer_call_fn_19567
*__inference_dropout_11_layer_call_fn_19572?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_11_layer_call_and_return_conditional_losses_19577
E__inference_dropout_11_layer_call_and_return_conditional_losses_19589?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dense_27_layer_call_fn_19598?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_27_layer_call_and_return_conditional_losses_19608?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_18181input_17"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_sequential_3_layer_call_fn_16862
,__inference_sequential_3_layer_call_fn_19621
,__inference_sequential_3_layer_call_fn_19634
,__inference_sequential_3_layer_call_fn_16935?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_sequential_3_layer_call_and_return_conditional_losses_19691
G__inference_sequential_3_layer_call_and_return_conditional_losses_19748
G__inference_sequential_3_layer_call_and_return_conditional_losses_16949
G__inference_sequential_3_layer_call_and_return_conditional_losses_16963?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_24_layer_call_fn_19757?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_24_layer_call_and_return_conditional_losses_19788?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_25_layer_call_fn_19797?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_25_layer_call_and_return_conditional_losses_19827?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_16770?=<>?@ABCDEJKFGHILM'(126?3
,?)
'?$
input_17??????????'
? "3?0
.
dense_27"?
dense_27??????????
C__inference_dense_24_layer_call_and_return_conditional_losses_19788fFG4?1
*?'
%?"
inputs??????????'
? "*?'
 ?
0??????????'
? ?
(__inference_dense_24_layer_call_fn_19757YFG4?1
*?'
%?"
inputs??????????'
? "???????????'?
C__inference_dense_25_layer_call_and_return_conditional_losses_19827fHI4?1
*?'
%?"
inputs??????????'
? "*?'
 ?
0??????????'
? ?
(__inference_dense_25_layer_call_fn_19797YHI4?1
*?'
%?"
inputs??????????'
? "???????????'?
C__inference_dense_26_layer_call_and_return_conditional_losses_19562\'(/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? {
(__inference_dense_26_layer_call_fn_19551O'(/?,
%?"
 ?
inputs?????????
? "???????????
C__inference_dense_27_layer_call_and_return_conditional_losses_19608\12/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? {
(__inference_dense_27_layer_call_fn_19598O12/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_dropout_10_layer_call_and_return_conditional_losses_19530\3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
E__inference_dropout_10_layer_call_and_return_conditional_losses_19542\3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? }
*__inference_dropout_10_layer_call_fn_19520O3?0
)?&
 ?
inputs?????????
p 
? "??????????}
*__inference_dropout_10_layer_call_fn_19525O3?0
)?&
 ?
inputs?????????
p
? "???????????
E__inference_dropout_11_layer_call_and_return_conditional_losses_19577\3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
E__inference_dropout_11_layer_call_and_return_conditional_losses_19589\3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? }
*__inference_dropout_11_layer_call_fn_19567O3?0
)?&
 ?
inputs?????????
p 
? "??????????}
*__inference_dropout_11_layer_call_fn_19572O3?0
)?&
 ?
inputs?????????
p
? "???????????
U__inference_global_average_pooling1d_3_layer_call_and_return_conditional_losses_19509{I?F
??<
6?3
inputs'???????????????????????????

 
? ".?+
$?!
0??????????????????
? ?
U__inference_global_average_pooling1d_3_layer_call_and_return_conditional_losses_19515a8?5
.?+
%?"
inputs??????????'

 
? "%?"
?
0?????????
? ?
:__inference_global_average_pooling1d_3_layer_call_fn_19498nI?F
??<
6?3
inputs'???????????????????????????

 
? "!????????????????????
:__inference_global_average_pooling1d_3_layer_call_fn_19503T8?5
.?+
%?"
inputs??????????'

 
? "???????????
B__inference_model_1_layer_call_and_return_conditional_losses_18069=<>?@ABCDEJKFGHILM'(12>?;
4?1
'?$
input_17??????????'
p 

 
? "%?"
?
0?????????
? ?
B__inference_model_1_layer_call_and_return_conditional_losses_18124=<>?@ABCDEJKFGHILM'(12>?;
4?1
'?$
input_17??????????'
p

 
? "%?"
?
0?????????
? ?
B__inference_model_1_layer_call_and_return_conditional_losses_18565}=<>?@ABCDEJKFGHILM'(12<?9
2?/
%?"
inputs??????????'
p 

 
? "%?"
?
0?????????
? ?
B__inference_model_1_layer_call_and_return_conditional_losses_18879}=<>?@ABCDEJKFGHILM'(12<?9
2?/
%?"
inputs??????????'
p

 
? "%?"
?
0?????????
? ?
'__inference_model_1_layer_call_fn_17398r=<>?@ABCDEJKFGHILM'(12>?;
4?1
'?$
input_17??????????'
p 

 
? "???????????
'__inference_model_1_layer_call_fn_18014r=<>?@ABCDEJKFGHILM'(12>?;
4?1
'?$
input_17??????????'
p

 
? "???????????
'__inference_model_1_layer_call_fn_18230p=<>?@ABCDEJKFGHILM'(12<?9
2?/
%?"
inputs??????????'
p 

 
? "???????????
'__inference_model_1_layer_call_fn_18279p=<>?@ABCDEJKFGHILM'(12<?9
2?/
%?"
inputs??????????'
p

 
? "???????????
G__inference_sequential_3_layer_call_and_return_conditional_losses_16949xFGHID?A
:?7
-?*
dense_24_input??????????'
p 

 
? "*?'
 ?
0??????????'
? ?
G__inference_sequential_3_layer_call_and_return_conditional_losses_16963xFGHID?A
:?7
-?*
dense_24_input??????????'
p

 
? "*?'
 ?
0??????????'
? ?
G__inference_sequential_3_layer_call_and_return_conditional_losses_19691pFGHI<?9
2?/
%?"
inputs??????????'
p 

 
? "*?'
 ?
0??????????'
? ?
G__inference_sequential_3_layer_call_and_return_conditional_losses_19748pFGHI<?9
2?/
%?"
inputs??????????'
p

 
? "*?'
 ?
0??????????'
? ?
,__inference_sequential_3_layer_call_fn_16862kFGHID?A
:?7
-?*
dense_24_input??????????'
p 

 
? "???????????'?
,__inference_sequential_3_layer_call_fn_16935kFGHID?A
:?7
-?*
dense_24_input??????????'
p

 
? "???????????'?
,__inference_sequential_3_layer_call_fn_19621cFGHI<?9
2?/
%?"
inputs??????????'
p 

 
? "???????????'?
,__inference_sequential_3_layer_call_fn_19634cFGHI<?9
2?/
%?"
inputs??????????'
p

 
? "???????????'?
#__inference_signature_wrapper_18181?=<>?@ABCDEJKFGHILM'(12B??
? 
8?5
3
input_17'?$
input_17??????????'"3?0
.
dense_27"?
dense_27??????????
Z__inference_token_and_position_embedding_14_layer_call_and_return_conditional_losses_18917a=</?,
%?"
 ?
x??????????'
? "*?'
 ?
0??????????'
? ?
?__inference_token_and_position_embedding_14_layer_call_fn_18888T=</?,
%?"
 ?
x??????????'
? "???????????'?
N__inference_transformer_block_3_layer_call_and_return_conditional_losses_19235x>?@ABCDEJKFGHILM8?5
.?+
%?"
inputs??????????'
p 
? "*?'
 ?
0??????????'
? ?
N__inference_transformer_block_3_layer_call_and_return_conditional_losses_19493x>?@ABCDEJKFGHILM8?5
.?+
%?"
inputs??????????'
p
? "*?'
 ?
0??????????'
? ?
3__inference_transformer_block_3_layer_call_fn_18954k>?@ABCDEJKFGHILM8?5
.?+
%?"
inputs??????????'
p 
? "???????????'?
3__inference_transformer_block_3_layer_call_fn_18991k>?@ABCDEJKFGHILM8?5
.?+
%?"
inputs??????????'
p
? "???????????'