#!/usr/bin/perl

use Data::Dumper;

my $file = shift;
my $file2 = shift;

my $data;                                                                                                                                                                                                      
my $elim = {};                                                                                                                                                                                                 
                                                                                                                                                                                                               
open(G,$file2);                                                                                                                                                                                                
while(<G>){                                                                                                                                                                                                    
        my @a = split(' ',$_);                                                                                                                                                                                 
        chomp($a[8]);                                                                                                                                                                                          
        $data->{$a[0]} = {prot=>$a[1],cov=>$a[2],ident=>$a[3],type=>$a[4],start=>$a[5],end=>$a[6],strand=>$a[7],db=>$a[8]};                                                                                    
}                                                                                                                                                                                                              
close(G);                                                                                                                                                                                                      
                                                                                                                                                                                                               
my $o;                                                                                                                                                                                                         
my $u;                                                                                                                                                                                                         
                                                                                                                                                                                                               
open(F,$file);                                                                                                                                                                                                 
while(<F>){                                                                                                                                                                                                    
        my @a;                                                                                                                                                                                                 
        if (/(\S+)\s+(\S+).+/){                                                                                                                                                                                
                $o = $1;                                                                                                                                                                                       
                $u = $2;
        }
        if ($o != $u){
                if (!(defined $elim->{$o}) && !(defined $elim->{$u})){
                        my $e = getElim($o,$data->{$o},$u,$data->{$u});
if ($e == 1){
#print "OOOOOOOOOOOOOOOOOOOOOO $o : ". $data->{$o}->{ident} ."-".$data->{$o}->{cov}."\n";
#print "OOOOOOOOOOOOOOOOOOOOOO $u : ". $data->{$u}->{ident} ."-".$data->{$u}->{cov}."\n";
}
                        $elim->{$e} = 1;
                }
        }

}
close(F);

print getResults($data,$elim);

sub getElim {
        my $a1 = shift;
        my $da1 = shift;
        my $a2 = shift;
        my $da2 = shift;
if ($da1->{ident} > 90.0 && $da2->{ident} > 90.0){
        if ($da1->{cov} >= $da2->{cov}) {
                return $a2;
        } else {
                return $a1;
        }
} else {
        if ($da1->{ident} > $da2->{ident}){
                return $a2;
        } elsif ($da2->{ident} > $da1->{ident}) {
                return $a1;
        } else {
                if ($da1->{cov} >= $da2->{cov}) {
                        return $a2;
                } else {
                        return $a1;
                }
        }
}
}

sub getResults {
        my $data = shift;
        my $elim = shift;
        foreach my $k (keys %{$data}) {
                if (!defined $elim->{$k}){
                       # print $data->{$k}->{cov} . "\t" . $data->{$k}->{ident} . "\t" . $data->{$k}->{prot} . "\t";
                       # print $data->{$k}->{type} . "\t" 
			print $data->{$k}->{start} . "\t" . $data->{$k}->{end};
                        print "\t" . $data->{$k}->{strand}."\t".$data->{$k}->{prot}."\n";# . "\t" . $data->{$k}->{db} . "\n";
                        #print "YES $k\n";
                }
        }
}

