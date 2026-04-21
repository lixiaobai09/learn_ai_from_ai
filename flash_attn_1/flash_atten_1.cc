#include <iostream>
#include <cmath>

constexpr int N = 1024;
constexpr int dim = 8;
// for Q block size
constexpr int Br = 16;
// for K,V block size
constexpr int Bc = dim;
constexpr float INF_NEG = std::numeric_limits<float>::lowest();

float Q[N][dim];
float K[N][dim];
float V[N][dim];
float O[N][dim];

// output = softmax(input)
template<int M, int D>
void softmax(const float input[M][D],
             float output[M][D],
             float l[M],
             float m[M]) {
  for (int i = 0; i < M; ++i) {
    const float* row = input[i];
    float max_val = row[0];
    for (int j = 1; j < D; ++j) {
      if (row[j] > max_val) {
        max_val = row[j];
      }
    }
    float sum = 0.0f;
    for (int j = 0; j < D; ++j) {
      sum += std::exp(row[j] - max_val);
    }
    for (int j = 0; j < D; ++j) {
      output[i][j] = std::exp(row[j] - max_val) / sum;
    }
    l[i] = sum;
    m[i] = max_val;
  }
}

// output = A * B
template<int M, int N, int D>
void matmul_transpose(const float A[M][D],
                      const float B[N][D],
                      float output[M][N]) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < D; ++k) {
        sum += A[i][k] * B[j][k];
      }
      output[i][j] = sum;
    }
  }
}

template<int M, int N>
void random_fill(float matrix[M][N]) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      // Random float between 0.0 and 10.0
      matrix[i][j] = static_cast<float>(rand() % 100) / 10.0f;
    }
  }
}

template<int M>
void fill_elements(float array[M], float value) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
        array[i] = value;
    }
  }
}

template<int M, int N>
void copy_matrix(const float src[M][N], float dst[M][N]) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      dst[i][j] = src[i][j];
    }
  }
}

void init() {
  srand(42); // Seed for reproducibility
  random_fill<N, dim>(Q);
  random_fill<N, dim>(K);
  random_fill<N, dim>(V);
}

int main(int argc, char** argv) {
  init();
  float S[N][N];
  float P[N][N];
  float O_1[N][dim];
  float l[N];
  float m[N];
  // original attention: output = softmax(Q * K^T) * V
  // Step 1: S = Q * K^T
  matmul_transpose<N, N, dim>(Q, K, S);
  // Step 2: P = softmax(S)
  softmax<N, N>(S, P, l, m);
  // Step 3: output = P * V
  matmul_transpose<N, dim, N>(P, V, O);
  copy_matrix<N, dim>(O, O_1);


  fill_elements<Tr>(l, 0);
  fill_elements<Tr>(m, INF_NEG);
  int Tc = N / Bc;
  int Tr = N / Br;
  for (int i = 0; i < Tc; ++i) {
    float (*k_block)[dim] = K + i * Bc;
    float (*v_block)[dim] = V + i * Bc;
    for (int j = 0; j < Tr; ++j) {
      float (*q_block)[dim] = Q + j * Br;
      float (*o_block)[dim] = O + j * Br;
      float *l_block = l + j * Br;
      float *m_block = m + j * Br;
      // step 1: s = q_block * k_block^T
      float s[Br][Bc];
      matmul_transpose<Br, Bc, dim>(q_block, k_block, s);
      // step 2: s_out = softmax(s), and get max_s and sum_exp_s
      float l_block_local[Br];
      float m_block_local[Br];
      float s_out[Br][Bc];
      softmax<Br, Bc>(s, s_out, l_block_local, m_block_local);
      // step 3: m_new = max(m_block, m_block_local),
      //         l_new = l_block * exp(m_block - m_new) + l_block_local * exp(m_block_local - m_new)
      float l_new[Br];
      float m_new[Br];
      for (int k = 0; k < Br; ++k) {
        m_new[k] = std::max(m_block[k], m_block_local[k]);
        l_new[k] = (l_block[k] * std::exp(m_block[k] - m_new[k])
                    + l_block_local[k] * std::exp(m_block_local[k] - m_new[k]));
      }
      // step 4: o_block_new = (o_block * exp(m_block - m_new) * l_block + s_out * v_block * exp(m_block_local - m_new) * l_block_local) / l_new
      float o_block_new[Br][dim];
      for (int k = 0; k < Br; ++k) {
        for (int d = 0; d < dim; ++d) {
          float o_old = o_block[k][d] * std::exp(m_block[k] - m_new[k]) * l_block[k];
          float o_new = 0.0f;
          for (int c = 0; c < Bc; ++c) {
            o_new += s_out[k][c] * v_block[c][d];
          }
          o_new = o_new * std::exp(m_block_local[k] - m_new[k]) * l_block_local[k];
          o_block_new[k][d] = (o_old + o_new) / l_new[k];
        }
      }
      // step 5: copy o_block_new to o_block, and copy l_new, m_new to l_block, m_block
      for (int k = 0; k < Br; ++k) {
        for (int d = 0; d < dim; ++d) {
          o_block[k][d] = o_block_new[k][d];
        }
        l_block[k] = l_new[k];
        m_block[k] = m_new[k];
      }
    }
  }
  return 0;
}
