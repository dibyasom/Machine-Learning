#include <stdio.h>


int add(int num1, int num2)
{
    int sum;
    sum= num1+num2;
    return sum;
}
int sub(int a, int b)
{
    int diff;
    diff= a-b;
    return diff;
}
int main ()
{
    int a,b;
    printf ("enter first number\n");
    scanf("%d", &a);
    printf("Enter second number\n");
    scanf("%d", &b);

    int addition,substraction;
    addition = add (a,b);
    substraction= sub (a,b);
    printf("Addition: %d\n", addition);
    printf ("Substraction: %d\n", substraction);
    
    return 0;
}