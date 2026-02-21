#!/usr/bin/env python3
"""
scripts/row_autoencoder.py

Simple tabular autoencoder that:
- loads a CSV where each row is a sample
- selects numeric columns, scales them, trains a Dense autoencoder
- encodes the whole dataset into a latent representation
- saves latent features to a CSV, preserving non-numeric columns (like IDs)

Usage examples:
  python scripts/row_autoencoder.py --input Data/X_train_val_processed.csv --output Data/X_train_latent_rows.csv --latent-dim 16 --epochs 50

You can also skip training and only encode when providing `--load-model path/to/encoder.h5`.
"""
import argparse
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_dense_autoencoder(input_dim, latent_dim):
    inp = keras.Input(shape=(input_dim,), name="input")
    x = layers.Dense(max(64, input_dim * 2), activation="relu")(inp)
    x = layers.Dense(max(32, input_dim), activation="relu")(x)
    z = layers.Dense(latent_dim, activation="linear", name="bottleneck")(x)

    # decoder
    x = layers.Dense(max(32, input_dim), activation="relu")(z)
    x = layers.Dense(max(64, input_dim * 2), activation="relu")(x)
    out = layers.Dense(input_dim, activation="linear", name="reconstruction")(x)

    auto = keras.Model(inputs=inp, outputs=out, name="dense_autoencoder")
    encoder = keras.Model(inputs=inp, outputs=z, name="encoder")
    return auto, encoder


def _derive_output_path(inp_path):
    base, ext = os.path.splitext(inp_path)
    return base + "_out" + ext


def _ensure_numeric_cols(df, numeric_cols):
    # Return df[numeric_cols] in the same column order. If some columns are missing, fill with zeros.
    cols_present = [c for c in numeric_cols if c in df.columns]
    cols_missing = [c for c in numeric_cols if c not in df.columns]
    if cols_missing:
        # create zeros for missing
        zeros = np.zeros((len(df), len(cols_missing)), dtype=float)
        df_missing = pd.DataFrame(zeros, columns=cols_missing, index=df.index)
        df_num = pd.concat([df[cols_present], df_missing], axis=1)[numeric_cols]
        print(f"Warning: missing numeric cols in {getattr(df, 'name', 'input')}: {cols_missing}; filled with zeros")
        return df_num
    else:
        return df[numeric_cols]


def encode_csv(path, encoder, scaler, numeric_cols, output_path=None, batch_size=128):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    # preserve non-numeric cols
    num_df = _ensure_numeric_cols(df, numeric_cols)
    X = num_df.values.astype(np.float32)
    Xs = scaler.transform(X)
    Z = encoder.predict(Xs, batch_size=batch_size)

    latent_cols = [f"latent_{i}" for i in range(Z.shape[1])]
    df_latent = pd.DataFrame(Z, columns=latent_cols, index=df.index)
    other_df = df.loc[:, df.columns.difference(num_df.columns)]
    out_df = pd.concat([other_df.reset_index(drop=True), df_latent.reset_index(drop=True)], axis=1)

    if output_path is None:
        output_path = _derive_output_path(path)
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Saved encoded CSV to {output_path} (shape {out_df.shape})")
    return output_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input CSV path (training / primary file)")
    p.add_argument("--output", required=False, default=None, help="Output latent CSV path for primary input (if omitted adds '_out' suffix to input file)")
    p.add_argument("--out-dir", required=False, default="../DataTemp", help="Directory to place output latent files and model artifacts (default ../DataTemp)")
    p.add_argument("--input-val", required=False, default=None, help="Optional validation CSV to encode after training/loading")
    p.add_argument("--input-test", required=False, default=None, help="Optional test CSV to encode after training/loading")
    p.add_argument("--output-val", required=False, default=None, help="Optional output path for validation encoded CSV (if omitted will be derived)")
    p.add_argument("--output-test", required=False, default=None, help="Optional output path for test encoded CSV (if omitted will be derived)")
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--test-size", type=float, default=0.1)
    p.add_argument("--save-model", action="store_true", help="Save encoder, scaler and numeric cols next to output file")
    p.add_argument("--load-model", default=None, help="If provided, skip training and load encoder model (.h5) to encode input and optional val/test")
    p.add_argument("--scaler-path", default=None, help="Path to scaler joblib to load (useful with --load-model)")
    args = p.parse_args()

    inp = args.input
    out = args.output

    # derive output into out_dir when an explicit output is not provided
    base = os.path.dirname(__file__)
    out_dir_arg = args.out_dir or "../DataTemp"
    out_dir_abs = os.path.abspath(os.path.join(base, out_dir_arg))
    os.makedirs(out_dir_abs, exist_ok=True)
    if out is None:
        derived = _derive_output_path(inp)
        out = os.path.join(out_dir_abs, os.path.basename(derived))

    if not os.path.exists(inp):
        raise FileNotFoundError(f"Input file not found: {inp}")

    df = pd.read_csv(inp)
    print(f"Loaded CSV with shape {df.shape}")

    # identify numeric columns used for training/encoding
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    # Exclude `sample_index` from encoder features (keep it for preservation in outputs)
    if "sample_index" in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=["sample_index"])
    if numeric_df.shape[1] == 0:
        raise ValueError("No numeric columns found in the input CSV. Autoencoder needs numeric features.")
    numeric_cols = list(numeric_df.columns)
    print(f"Numeric columns used: {numeric_cols}")

    X = numeric_df.values.astype(np.float32)
    print(f"Numeric feature matrix shape: {X.shape}")

    scaler = None
    encoder = None

    # If loading a model, prefer that path; scaler optional
    if args.load_model:
        print(f"Loading encoder model from {args.load_model}")
        encoder = keras.models.load_model(args.load_model, compile=False)
        if args.scaler_path:
            print(f"Loading scaler from {args.scaler_path}")
            scaler = joblib.load(args.scaler_path)
        else:
            print("No scaler provided; fitting a new StandardScaler on the input data")
            scaler = StandardScaler().fit(X)
        # encode primary and any additional files
        encode_csv(inp, encoder, scaler, numeric_cols, output_path=out, batch_size=args.batch_size)

        if args.input_val:
            out_val = args.output_val or os.path.join(out_dir_abs, os.path.basename(_derive_output_path(args.input_val)))
            encode_csv(args.input_val, encoder, scaler, numeric_cols, output_path=out_val, batch_size=args.batch_size)
        if args.input_test:
            out_test = args.output_test or os.path.join(out_dir_abs, os.path.basename(_derive_output_path(args.input_test)))
            encode_csv(args.input_test, encoder, scaler, numeric_cols, output_path=out_test, batch_size=args.batch_size)

        return

    # Train a new autoencoder on primary input
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    input_dim = Xs.shape[1]
    auto, encoder = build_dense_autoencoder(input_dim, args.latent_dim)
    auto.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    auto.summary()

    Xtrain, Xval = train_test_split(Xs, test_size=args.test_size, random_state=42)
    callbacks = [
        keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
    ]

    auto.fit(
        Xtrain,
        Xtrain,
        validation_data=(Xval, Xval),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    # After training, encode primary and optional val/test
    encode_csv(inp, encoder, scaler, numeric_cols, output_path=out, batch_size=args.batch_size)

    if args.input_val:
        out_val = args.output_val or os.path.join(out_dir_abs, os.path.basename(_derive_output_path(args.input_val)))
        encode_csv(args.input_val, encoder, scaler, numeric_cols, output_path=out_val, batch_size=args.batch_size)

    if args.input_test:
        out_test = args.output_test or os.path.join(out_dir_abs, os.path.basename(_derive_output_path(args.input_test)))
        encode_csv(args.input_test, encoder, scaler, numeric_cols, output_path=out_test, batch_size=args.batch_size)

    if args.save_model:
        # save encoder and scaler and numeric_cols next to primary output
        base = os.path.splitext(out)[0]
        encoder_path = base + "_encoder.h5"
        scaler_path = base + "_scaler.joblib"
        cols_path = base + "_numeric_cols.joblib"
        print(f"Saving encoder to {encoder_path}, scaler to {scaler_path}, numeric cols to {cols_path}")
        encoder.save(encoder_path, include_optimizer=False)
        joblib.dump(scaler, scaler_path)
        joblib.dump(numeric_cols, cols_path)


if __name__ == "__main__":
    main()
