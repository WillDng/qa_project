import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

import classification_lib as cl

def do_and_stack_cosine(
    cosine_tfidftransformer,
    df_to_stack_to,
    df,
    norm=False
):
    title_cosine_transformed = cosine_tfidftransformer.transform(df.question_title)
    question_cosine_transformed = cosine_tfidftransformer.transform(df.question_body)
    answer_cosine_transformed = cosine_tfidftransformer.transform(df.answer)

    title_question_similarity = cl.format_cosine(
        cosine_similarity(
            title_cosine_transformed, 
            question_cosine_transformed
        ).diagonal()
    )
    title_answer_similarity = cl.format_cosine(
        cosine_similarity(
            title_cosine_transformed, 
            answer_cosine_transformed
        ).diagonal()
    )
    question_answer_similarity = cl.format_cosine(
        cosine_similarity(
            question_cosine_transformed, 
            answer_cosine_transformed,
            
        ).diagonal()
    )

    df_transformed = np.hstack(
        (
            df_to_stack_to,
            title_question_similarity,
            title_answer_similarity,
            question_answer_similarity
        )
    )

    #tu peux mettre la normalisation ici
    # if norm:
    #     df_transformed = blabla(df_transformed)

    return df_transformed

