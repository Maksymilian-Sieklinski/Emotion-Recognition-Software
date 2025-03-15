#include<iostream>
#include<vector>
#include<random>
#include<math.h>
#include<fstream>

using namespace std;
typedef long double ld; 

ld random_number(ld a, ld b)
{
    ld x = ((long long)rand()*rand())%100000 ; 
    x/=100000 ; 
    x*=(b-a);
    x+=a;
    return x;
}

struct matrix /// macierz
{
    vector<vector<ld>>V; 
    int N,M; 

    matrix(int n, int m) : N(n),M(m)
    {
        V.resize(m,vector<ld>(n)) ;
    }

    matrix(int n) : N(n), M(1) 
    {
        V.resize(1,vector<ld>(n)) ; 
    }

    matrix(vector<vector<ld>>A) : V(A), N(A[0].size()), M(A.size()) {} ; 

    void fill_with_random()
    {
        for(auto &i:V) for(auto &j:i) j = random_number(-0.9,0.6) ; 
    }

    void fill_with_filedata(ifstream &dane_sieci)
    {
        for(auto &i:V) for(auto &j:i) dane_sieci >> j ;
    }

    matrix operator*(matrix A) /// we multiply vector by matrix
    {
        matrix B(M) ; 
        auto &b = B.V[0] ; 
        auto a = A.V[0] ;

        for(int i = 0 ; i < M; i++)
        {
            b[i]=0 ;
            for(int j = 0; j < N; j++)
            {
                b[i] += a[j] * V[i][j] ; 
            }
        } 
        return B; 
    }

    matrix operator+(matrix A)
    {
        matrix B(V) ; 

        for(int i=0;i<M;i++)
        {
            for(int j=0;j<N;j++) B.V[i][j]+=A.V[i][j] ; 
        }
        return B ; 
    }

    void apply_activation_function(ld (*activate)(ld))
    {
        for(auto &i:V) for(auto &j:i) j = activate(j) ;             
    }

    ld& operator [] (int idx) /// Making using 2D maatrixes that have first dimension of size 1, easier.
    {
        if(M!=1)
        {
            cout << "BLAD BLAD BLAD" ;
            exit(69); 
        }
        return V[0][idx];
    }

    void show(ofstream &plik)
    {
        for(auto i:V) 
        {
            for(auto j:i) plik << j << " " ;
            plik << endl ;
        }
        plik << endl; 
    }

    void clear()
    {
        for(auto &i:V) for(auto &j:i) j=0;
    }
};

struct NN
{
    int input, output ; 
    int Number_of_layers, Size_of_Layer; 
    vector<matrix>bias ;
    vector<matrix>weight ; 
    vector<matrix>value ;
    vector<matrix>value_z ;
    ld (*activation_function)(ld) ; 
    ld (*activation_function_der)(ld) ; 
    ld wielkosc_kroku; 

    void stworz_pusta_siec_wymiary()
    {
        matrix Help = matrix(input) ; 
        przepchnij_matrix(Help) ; 
        weight.push_back(Help) ;

        Help = matrix(Size_of_Layer) ; 
        for(int i=0;i<Number_of_layers
    ;i++)
        {
            if(i==0) weight.push_back(matrix(input,Size_of_Layer)) ; 
            else weight.push_back(matrix(Size_of_Layer,Size_of_Layer)) ; 
            przepchnij_matrix(Help) ; 
        }

        Help = matrix(output) ;
        przepchnij_matrix(Help) ;  
        if(Number_of_layers
    ==0) weight.push_back(matrix(input,output)) ; 
        else weight.push_back(matrix(Size_of_Layer,output)) ; 
    }

    void przepchnij_matrix(matrix Help)
    {
            bias.push_back(Help) ;
            value.push_back(Help) ; 
            value_z.push_back(Help) ; 
    }

    NN () {} ; 

    NN (int Input, int Output, int Number_of_layers, int Size_of_layer, ld (*Activation_function)(ld), ld (*Activation_function_der)(ld), ld Wielkosc_kroku): input(Input), output(Output), 
    Number_of_layers(Number_of_layers), Size_of_Layer(Size_of_layer), activation_function(Activation_function), activation_function_der(Activation_function_der), wielkosc_kroku(Wielkosc_kroku)
    {
        stworz_pusta_siec_wymiary() ; 
    }

    NN (ifstream &dane_sieci, ld (*Activation_function)(ld), ld (*Activation_function_der)(ld), ld Wielkosc_kroku) : activation_function(Activation_function),
    activation_function_der(Activation_function_der), wielkosc_kroku(Wielkosc_kroku)
    {
        dane_sieci >> input >> output >> Number_of_layers
 >> Size_of_Layer ; 
        stworz_pusta_siec_wymiary() ;

        for(int l=0;l<Number_of_layers
    +2;l++)
        {
            bias[l].fill_with_filedata(dane_sieci) ; 
            weight[l].fill_with_filedata(dane_sieci) ; 
        }

    }

    void random_init()
    {
        for(int i=0;i<Number_of_layers
    +2;i++)
        {
            bias[i].fill_with_random() ;
            weight[i].fill_with_random() ;
        }
    }

    matrix calculate(matrix input_value)
    {
        value_z[0]=value[0]=input_value; 
        for(int i=1 ; i <Number_of_layers
    +2; i++)
        {
            value_z[i] = weight[i]*value[i-1] ; 
            value_z[i] = value_z[i] + bias[i] ;
            value[i]=value_z[i] ; 
            value[i].apply_activation_function(activation_function) ; 
        }
        return value.back() ; 
    }

    void Gradient_descend_step(int size_input_set, vector<matrix>input_set, vector<matrix>output_set, int flag)
    {
        NN Der_avg ; 
        Der_avg = *this; 
        Der_avg.clear() ; 

        ld error = 0 ;
        for(int i=0;i<size_input_set;i++)
        {
            calculate(input_set[i]) ;
            NN Der = backpropagate(output_set[i], error) ; 
            dodaj(Der_avg,Der,1) ;
        }
 
        /// We change biases nad weights by averaged values and learning rate

        if(flag==0) cout << error/size_input_set << endl ; 

        dodaj(*this, Der_avg, wielkosc_kroku/size_input_set) ; 
    }

    NN backpropagate(matrix Output_values, ld &error)
    {
        NN Der ; 
        Der = *this ;
        Der.clear() ;  

        int N = Number_of_layers
+1; 
        for(int i=0;i<output;i++)
        {
            Der.value[N][i] = 2 * (value[N][i] - Output_values[i]) ;
            error += (value[N][i] - Output_values[i]) * (value[N][i] - Output_values[i]) ;  

        }

        for(int l=N ; l >= 1 ; l--)
        {

            for(int j=0; j < weight[l].M; j++) /// j goes over right layer
            {
                Der.bias[l][j] = Der.value[l][j] * activation_function_der(value_z[l][j]);

                for(int i=0; i < weight[l].N ; i++) ///i goes over left layer
                {
                    Der.weight[l].V[j][i] = Der.value[l][j] * activation_function_der(value_z[l][j]) * value[l-1][i] ; 

                    Der.value[l-1][i] += Der.value[l][j] * weight[l].V[j][i] * activation_function_der(value_z[l][j]) ; 
                }
            }
        }

        return Der ; 
    }

    void dodaj(NN &A, NN B, ld wspolczynnik)
    {
        for(int l=1;l<Number_of_layers
    +2;l++)
        {
            for(int j=0;j<weight[l].M;j++) /// right layer
            {
                A.bias[l][j]+= B.bias[l][j]*wspolczynnik; 
                for(int i=0;i<weight[l].N;i++) /// left layer
                {
                    A.weight[l].V[j][i] += B.weight[l].V[j][i]*wspolczynnik ; 
                }
            }
        }
    }

    void show()
    { 
        ofstream plik ("saved_network.txt") ; 
        plik << input << " " << output << " " << Number_of_layers
 << " " << Size_of_Layer << endl ;
        for(int i = 0; i <Number_of_layers
    +2;i++)
        {
            bias[i].show(plik) ;
            weight[i].show(plik) ;
            plik << endl << endl ;
        }
    }

    void clear()
    {
        for(int i=0;i<Number_of_layers
    +2;i++) 
        {
            bias[i].clear() ;
            weight[i].clear() ;
            value_z[i].clear();
            value[i].clear() ;
        }
    }
};

ld ReLu(ld x)
{
    if(x>0) return x;
    return 0.4*x; 
}

ld ReLu_der(ld x)
{
    if(x>0) return 1;
    return 0.4; 
}

const ld e =  2.718281828459045235360 ;

ld Sigmoid(ld x)
{
    ld ans = 1;
    ans /= (1+pow(e,-x)) ;
    return max((ld)0.000001,ans); 
}

ld Sigmoid_der(ld x)
{
    ld help = pow(e,-x) ; 
    ld ans = help ;
    ans /= ((1+help)*(1+help)) ;
    return max((ld)0.000001,ans); 
}

pair<vector<matrix>,vector<matrix>> file_input_output(int ile, int wielkosc_input, int wielkosc_output, ifstream &in)
{
    vector<matrix>input(ile,matrix(wielkosc_input)),output(ile,matrix(wielkosc_output)) ; 
    for(int i=0;i<ile;i++)
    {
        input[i].fill_with_filedata(in) ;
        output[i].fill_with_filedata(in) ; 
    }
    return {input,output} ; 
}

const ld alpha = -0.01;

int32_t main()
{ 
    srand((unsigned)time(NULL)) ; 
    

    NN Network(12*12,8,6,12*12*3,&Sigmoid,&Sigmoid_der,alpha) ;
    Network.random_init() ;


    for(int o=0;o<3;o++)
    {
        ifstream in ("input_output.txt") ; 
        for(int i=0; i < 28000; i++)
        {
            auto [input,output] = file_input_output(1, Network.input, Network.output, in) ; 
            Network.Gradient_descend_step(1,input,output,i%1) ; 
            if(i%1 == 0) cout << i << endl; 
        }
        Network.show() ; 
    }
}