# pyc_compiler

a custom C like script toy language compiler implemented by pure Python!

# example
``` c
    create_list(num){
        l = list();
        for(i=0; i<num;i++){
            if (len(l) > 10) break;
            l.append(i);
            }
        return l;
    }
    a = b = 4;
    c = create_list(1+a*5);
    if (a == 0) a+= 1;    
    b += -1;
    printf("%d %d %d\n", a, b, len(c));
```

# screenshot

![image](https://github.com/boywhp/pyc_compiler/blob/master/compiler.PNG)
