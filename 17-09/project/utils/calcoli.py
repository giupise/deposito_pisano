def somma(a: int, b: int) -> int:
    """
    Somma due numeri interi.

    Parameters
    ----------
    a : int
        Primo numero intero.
    b : int
        Secondo numero intero.

    Returns
    -------
    int
        La somma di due numeri.
    """
    return a + b

def conta_unici(lista):
    """
    Conta il numero di elementi unici in una lista.
    
    Parameters
    ----------
    lista : list or array-like
        Lista di elementi di cui contare i valori unici.
    
    Returns
    -------
    int
        Il numero di elementi unici nella lista.
        
    Examples
    --------
    >>> conta_unici([1, 2, 3, 2, 1])
    3
    >>> conta_unici(['a', 'b', 'a', 'c'])
    3
    >>> conta_unici([])
    0
    """
    return len(set(lista))

def primi_fino_a_n(n):
    """
    Genera tutti i numeri primi minori o uguali a n usando il metodo della divisione di prova.
    
    Parameters
    ----------
    n : int
        Limite superiore (incluso) fino al quale cercare i numeri primi.
        Deve essere un intero non negativo.
    
    Returns
    -------
    list of int
        Lista ordinata contenente tutti i numeri primi da 2 fino a n.
        Restituisce una lista vuota se n < 2.
        
    Examples
    --------
    >>> primi_fino_a_n(10)
    [2, 3, 5, 7]
    >>> primi_fino_a_n(20)
    [2, 3, 5, 7, 11, 13, 17, 19]
    >>> primi_fino_a_n(1)
    []
    >>> primi_fino_a_n(2)
    [2]
    """
    primi = []
    
    # Controlla ogni numero da 2 a n
    for numero in range(2, n + 1):
        # Assume che il numero sia primo
        e_primo = True
        
        # Controlla se è divisibile per qualsiasi numero da 2 a numero-1
        for divisore in range(2, numero):
            if numero % divisore == 0:
                # Se è divisibile, non è primo
                e_primo = False
                break
        
        # Se non ha trovato divisori, è primo
        if e_primo:
            primi.append(numero)
    
    return primi

