\version "2.12.0"

%% comandos

%% para pizzicato use \pizz, para arco \arco
pizz = ^\markup{\italic{pizz.}}
arco = ^\markup{\italic{arco}}

global = {
  %% aqui as configurações globais (andamento, compasso, etc)
  \key c \minor
  \time 3/2
  \repeat volta 2 {
    s1*3/2*31
    \mark \default
    s1*3/2*26
    \mark \default
    s1*3/2*22
  }
  \alternative {
    {
      %% casa 1:
      s1*3/2*3
    }
    %% casa 2:
    {
      s1*3/2
    }
  }
  s1*3/2*3
  \bar "||"
  \key c \major
  s1*3/2*25
  \bar "||"
  \key e \major
  s1*3/2*21
  \bar "||"
  \key c \minor
  s1*3/2*40
  \mark \default
  s1*3/2*24
  \mark \default
  s1*3/2*30
  \bar "||"
  \time 2/2
  s1*2/2*37
  \bar "|."
}

violinoum= { 
  \relative c' {

    c4. \p \< (d8 ees4. g8 c4. ees8) \!
    ees2 (fis,4) r8 \crescTextCresc g \< (c4. ees8)
    ees2 (aes,4) r8 a (c4. ees8)
    f2 (a,4) r8 bes (des4. f8)
    aes4. c,8 \! ees4. \< aes8 c4. ees,8 
    aes4. c8 ees4. aes,8 c4. ees8 \!
                                %7	
    aes4-. \f r4  g-. \f r4 r2  
    r1 s2
    <bes,,,-. ges'-.>4 \p r4 <a f'>4 r4 r2 
    r1 s2
    f''2 (des c)
    f2 ~ \times 2/3 {f4 \< e f \!} \times 2/3 {bes4 \> f des \!}
    f2 (des c)
    f2 ~ \times 2/3 {f4 \< e f \!} \times 2/3 {d'4 \> b f \!}	%15
    e1. \p 
    (f1.
    e1. 
    ees1. 
    d2) r2 r2
    r1 s2 
    r1 s2
                                %22
    r2 r2 r8 aes, g b
    c \< d ees fis g b c d ees g b c \!
    ees-. \f ees,-. ees-. ees'-. ees ees, ees ees' ees ees, ees ees'
    ees ees, ees ees' ees ees, ees ees' \crescTextCresc ees \< ees, ees ees'
    des des, des des' des des, des des' des des, des des'
                                %27 
    c ees, ees c' c ees, ees c' c ees, ees c' 
    ces ees, ees ces' ces ees, ees ces' ces ees, ees ces' 
    <ces-. ees,>4 r8 g, bes4. ees8 g4. bes,8
    ees4. ges8 bes4. ees,8 ges4. bes8 
    ees4-. \f r4 d4-. r4 r4 r8 bes
                                %32	
    ges'4-. r4 f-. r4 r2
    r1 s2
    r4 bes,-. \p (r4 ges-. r4 ees-.)
    r8 d d ees f2 \< ~ (f8 \> ges \!) ges f 
    r8 ees ees f ges2 \< ~ (ges8 \> bes \!) bes aes
                                %37	
    r4 ges \< (bes ees des ges) 
    f (ees des ees des ces) \!
    bes \f (ges) ees (ces) bes' (ges) 
    ees (ces) bes' \> \((f) ees (c)\) \!
    bes8-. \p (aes-.) aes-. d-. d-. ces-. ces-. f-. f-. d-. d-. aes'-.
                                %42
    aes d, d ges ges ees ees bes' bes ges ges ees'
    ees4 (fes-.) (r des-. r bes-.)
    r ees-. (r ces-. r aes-.)
    r8  a \p a gis fis2 ~ (fis8 eis) eis fis
    r8 a a gis fis2 ~ (fis8 eis) eis fis
                                %47
    r8 \crescTextCresc cis'\< cis b a2 ~ a8 gis gis a \!		
    r8 \< cis cis b b a a gis gis a a gis \!
    gis-. \f a-. a-. gis-. gis a a gis gis a a gis
    gis a a gis gis a a gis gis a a gis
    gis a a gis gis a a gis gis \< a a gis
                                %52
    gis a a gis gis a a gis gis a a bes \!
    << {<bes bes,>2. d,,8 f f bes bes d} \\ \stemUp {d,4 \f} >> \stemNeutral
    d'8 f f bes bes4. (ees,8)  aes4. (ees8)
    aes4. (d,8) g4. (ees8) ces4. (aes8)
    g4. (bes8) ges4. (bes8) f bes bes d
                                %57		
    d f f bes bes4 d,,8 f f bes bes d 
    d f f bes bes4 ees4.-> (c8) aes4 ~
    (aes8 f) d'4. (bes8) g4. (ees8) c'4 ~
    (c8 aes) c4. (aes8) f4. (c8) aes4 ~ 
    (aes8 f) ees4 r2 r2
                                %62
    r2 r2 c' \p 
    (bes2. a4 bes) ees
    (c bes2 a4 bes4) ees ~
    (ees8 d ees a,) ~ a bes (ees g ~ g fis \< g) d ~
    d ees (a c ~ c) a (bes ees ~ ees cis d) g ~ \!
                                %67
    g e (f d ees \> c aes ees c aes ees c) \! 
    b \p (c d c ees aes c ees) d (ees f ees 
    bes' g aes f ges ees c a ges ees d c)
    bes (a gis a c ees ges c) bes (a gis a
    ees' g, ees g) f' (ees d ees g ees g, ees)
                                %72
    f (ees d ees ees' g, ees) g (f' ees d ees 
    g ees g, ees) f \< (ees d ees bes'2) ~ \!
    bes4 \> (aes g2 \! f2) 
    ees r2 r2 
    r1 s2
    r2 r2 bes'\< ~ 
    (bes4 \! \>  a g2 \! f)
                                %79
    ees r2 r2 
    r1 s2
    r2 r g ~ 
    (g4 f ees2 d)
    r2 r aes' \p ~
    aes2. (b4 c2 ~ 
    c4 d e2. gis4
    a2. b4 c2) ~ \key c \major
                                %84	
    c8 \p (a c a \repeat "tremolo" 2 {c8 a8} \repeat "tremolo" 2 {c8 a8)} 
    \repeat "tremolo" 2 {c8 (a8} \repeat "tremolo" 2 {c8 a8} \repeat "tremolo" 2 {c8 a8)} 
    \repeat "tremolo" 2 {d8 (bes8)} \repeat "tremolo" 2 {bes8 (f8)} \repeat "tremolo" 2 {f8 (d8)}
    \repeat "tremolo" 2 {f8 (d8} \repeat "tremolo" 2 {e8 c8} \repeat "tremolo" 2 {d8 b8)}
    a4. \< (b8 c4. e8 a4. c8) \!
    \crescTextCresc c2 \< (ees,4) r4 r2
                                %90	       
    bes4. (c8 d4. f8 bes4. d8) 
    f2-. e-. <gis,-. b, e,> 
    <a-. c, e,>4 \! \f r e4. c8 a4 r4     
                                %adicionar o marc.
    a'4. b8 c4 r a,4. b8 
    c4 r4 r2 r2 
    <g'-. c, e,>4. e8 c2 << {<b f'>2} \\ \stemUp {d,4} >> \stemNeutral
                                %96
    r e'4-. \p (r des-. r bes-.)
    r a'4-. (r f-. r e-.)
    r8 dis dis  \< e fis2 \! \> ~ (fis8 c) \! c  b
    r8 a a  \< b c2 \! \> ~ (c8 a) \! a  fis
    r2 b4. \f g8 e4 r4
                                %adicionar o marc.
                                %101
    g'1  \f a2 
    b4 r r2 r
    d4. b8 g2 c
    r4 b4-. \p (r aes-. r f-.)
    r e'4-. (r c-. r b-.) 
    r8 ais ais  \< b cis2 \! \> ~ (cis8 g) \! g  fis
                                %107		
    r8 e e  \< fis g2 \! \> ~ (g8 e) \! e  cis
    R1*3/2
    r8 f f  \< g aes2 \! \> ~ (aes8 f) \! f  d
    R1*3/2
                                %111
    r8 fis fis  \< gis a2 \! \> ~ (a8 fis) \! fis  dis \key e \major
    R1*3/2	
    a4. \p (\crescTextCresc b8 \< cis4. e8 a4. cis8)
    r8 fis \f fis   gis a2 ~ (a8 fis)  fis  d 
                                %115		
    r8 bis bis cis dis2   ~ (dis8 a)  a  gis 
    r8 fis fis gis a2   ~ (a8 fis)  fis  d
    r8 bis bis cis dis2   ~ (dis8 a)  a  gis 
    r8 cis \f  cis bis r8 gis' gis fis fis d d cis
                                %119 
    r8 fis fis eis r8 cis' cis b b g g fis
    r8 a a gis r8 fis fis e r8 dis dis cis
    cis d d fis fis cis cis e e bis bis dis
    fis4. \f gis8 a2 r2
    b4. cis8 d2 r2
                                %124
    dis4. e8 eis4. fis8 fisis4. gis8
    gis2 ~ gis8 a a gis gis2 ~
    gis8 a  a gis gis2 ~ gis8 a a gis
    \crescTextCresc gis4. \< (a8) a4. (gis8) gis4. (a8)
    a4. \< (gis8) gis4. (b8) b4. (a8) 
                                %129
    a8-. \f fis-. fis-. e-. r4 a,,8 fis fis e r4
    a''8 e e d r4 a,8 e e d r4 
    a'''8 d, d cis r4 a,8 d, d cis r4 
    a'''8 cis, cis bis r4 a,8 cis, cis bis r4 \key ees \major
                                %133
    r2 c4. \p (d8 ees2)
    r2 c4. (d8 ees2)
    r2 c1 
    (d1 ees2) ~
    ees2 ~ ees4.  \< (aes8 c4. ees8 \!)
    \crescTextCresc ees2 \< (fis,4) r8 g (c4. ees8)
    ees2 (aes,4) r8 a8 (c4. ees8)
    
                                %140	
    f2 (a,4) r8 bes8 (des4. f8)
    aes4. c,8 ees4. aes8 c4. ees,8
    aes4. c8 ees4. aes,8 c4. ees8
    aes4-. \f r4 ees-. r4 r2
    R1*3/2
    <g,,-. b,>4 \p r4 <fis-. ais,>4 r4 r2	    	
    R1*3/2
    e4-. \pizz r4 d-. r4 r2 		
    R1*3/2
                                %149
    r4 fis r4 g r4 a
    r4 g r4 ees r4 d
    g'2 \arco (ees d)
                                %colocar espress. 			
    g ~ (\times 2/3 {g4 \< fis g \! } \times 2/3 {c4 \> g ees \!)} 
    g2 (ees d)
                                %colocar arco e espress. 			
    f ~ (\times 2/3 {f4 \< e f \! } \times 2/3 {d'4 \> b f \!)}
    e1. \p (
                                %156		
    f
    e
    \dimTextDim e,4. \> (f8 g2 e)
    f2  \> (a,1 \!)
    R1*3/2
    R1*3/2
    g'2 \p (\crescTextCresc e \< des)	
    c1 ~ (c8 des c e \<)
                                %161
    f g aes b c e f g aes b c f 
    d \f d, d d' d d, d d' c c, c c'
    ces ces, ces ces' ces ces, ces ces' bes \crescTextCresc bes, \< bes bes'
    bes bes, bes bes' bes bes, bes bes' ees ees, ees ees'
    des des, des des' des des, des des' des des, des des'
                                %169
    des des, des des' des des, des des' des des, des des'
    <des des,>4 r8 c,,8 ees4. aes8 c4. ees,8
    aes4. c8 ees4. aes,8 c4. ees8
    aes4-. r4 g-. r4 r4 r8 ees8
    c'4-. r4 b4-. r4 r2
    R1*3/2
                                %175    
    r4 g4-. \p ( r4 ees-. r4 c-.)
    r8 b b \< c d2 \> ~ (d8 ees \!) ees d
    r8 c c \< d ees2 \> ~ (ees8 g \!) g f
    r4 ees (g c \< bes ees)
    d (c bes c bes aes)
                                %180
    g \f (ees c aes) g'  
    (ees c aes)  g' \> (d c a)
    g8 \p (f)  f-. b-. b aes aes d d b b f'
    f b, b ees ees c c g' g ees ees c'
    c4 (des-.) (r bes-. r g-.)
                                %185
    r c-. (r aes-. r f-.)
    r8 ges \p ges f ees2 ~ (ees8 d) d ees
    r8 ges ges f ees2 ~ (ees8 d) d ees
    r8 \crescTextCresc bes' \< bes aes ges2 ~ (ges8 f) f ges
    r8 bes \< bes aes aes ges ges f f ges ges f \!
                                %190
    eis-. \f fis-. fis-. eis-. eis fis fis eis eis fis fis eis
    eis fis fis eis eis fis fis eis eis fis fis eis
    eis fis fis eis eis fis fis eis eis fis fis eis
    eis \< fis fis eis eis fis fis eis eis fis fis g \!
    << {<g b,>2. b,,8 d d g g b} \\ \stemUp {d,4 \f s4 s1} >> \stemNeutral
                                %atenção na linha acima
                                %195
    b'8 d d g g4. (c,8) f4. (c8)
    f4. (b,8) e4. (c8) aes4. (f8)
    e4. (g8) ees4. (g8) d g g b
    b d d g g4 b,,8 d d g g b
    b d d g g4 c4.-> (a8) f4 ~ 
                                %200
    (f8 d) b'4. (g8) e4. (c8) a'4 ~
    (a8 f) a4. (f8) d4. (a8) f4 ~
    (f8 d) c4 r2 r2
    r2 r2 a'2 \p 
                                %colocar o dolce
    (g2. fis4 g) c ~ 
                                %205	
    (c b c fis, g) c ~
    c8 (b c fis, ~ fis g c e ~ e dis e) b ~
    b (a fis' a ~ a \< fis g c ~ c ais b e) ~
    e (cis d b  c \p a f c a f d a)
    gis (a b a c f a c b c d c)
                                %210
    g' (e f d ees d a fis ees c b a)
    c ( ees fis a c ees fis a) g (fis eis fis
    c' e, c e) d' (c \< b c e c e, c \!)
    d (c b d c' e, c e) d' (c b c
    e c e, c) d (c b c g'2) ~ 
                                %215
    g4 \> (f e2 \! d)
    c2 r2 r2
    R1*3/2
    r2 r2 g'2 \< ~
    (g4 \> f \! e2 d2)
    c2 r2 r2
    R1*3/2
    r2 r2 g2 \p ~
                                %adicionar o piú		
    (g4 f ees2 d) \time 2/2
                                %224
    r4 \p des4-. (r4 bes4-.)
                                %adicionar crescendo ed agitato
    r4 aes'-. (r4 f4-.)
    r8 b b c d2 ~
    d8 \< ees ees f f g aes g
    r4 des4-. \p (\crescTextCresc r4 \< bes4-.)
    r4 aes'4-. (r4 f4-.)
    r8 b b c d2 ~
                                %231 
    d8 \< ees ees f f g aes g
    g2 \f g,
    c,4. d8 e g c e
    <a a,,>2 <c, a,>2
    f,4. g8 a c f a
    c4-. c \f (des c
    bes aes f des)
                                %238
    r4 c2 (ees,4)
    r4 aes'2 (b,4)
    r4 c (des c
    bes aes f des)
    r4 c2 (ees,4) 
    r4 aes'2 (b,4)
                                %244 ao 260 - À partir daqui cada linha representa os acordes ligados e estão escritos em "duas vozes".		
    r2 << {<c e,>2 ~ <c e,>4 b4 \rest 
            <c f,>2 ~ <c f,>4 b4 \rest 
            <des f,>2 ~ <des f,>4. ees8 \< f a des f \! e4 b,4 \rest
            <c e,>2 ~ <c e,>4 b4 \rest <c f,>2 ~ <c f,>4 b4 \rest 
            <des f,>2 ~ <des f,>4. ees8 \< f a des f \! e4 b,4 \rest 
            <c' e,>2 ~ \dimTextDim <c e,>4 \> b,4 \rest \!
            <g' c,>2 ~ <g c,>4 b,4 \rest 
            <c e,>2 ~ <c e,>4 b4 \rest 
            <g g,>2 ~ <g g,>4 b4 \rest 
            <e, g,>2 ~  <e g,>4 b'4 \rest 
            g,2 ~ g4 b'4 \rest  
            g,2 ~ g4 b'4 \rest  b2 \rest 
            << <c c'>1 ^\fermata { s4 \p s4\< s4\! \> s4\! } >> }
          \\ \stemUp { g,4 \fz s4 s2 
                       a4 \fz s4 s2 
                       aes4 \fz s4 s1 s2 
                       g4 \fz s4 s2 
                       a4 \fz s4 s2 
                       aes4 \fz s4 s1 s2 
                       g'4 \fz s4 s2 
                       e4 s4 s2 
                       g,4 s4 } >>
    
    
  }	
}

violinodois= { 
  \relative c' {
    
    r2 r2  c4. \p (ees8) 
    ees2 r2 \crescTextCresc c4. \< (ees8) \! 
    ees2 r2 c4. ees8
    f2 (a,4) r8 bes (des4. f8)
    aes4. c,8 ees4. \< aes8 c4. ees,8 
    aes4. c8 ees4. aes,8 c4. ees8 \!
                                %7	
    aes4-. \f r4  <g-. b, d,>4 \f r4 r2  
    r1 s2
    des,4-. \p r4 c4-. r4 r2 
    r1 s2
    r2 f4. \p (g8 aes4. f8)
    des'2  ~ (des4 \< c \! des \> bes \!)
    c r4 f,4. (g8 aes4. f8) 
    d'2 (cis4 \< d4 \! f \> b, \!)
                                %15
    c2 (aes g)
    c2 ( ~ \times 2/3 {c4 \< b c \!} \times 2/3 {f4 \> c aes ) \!}
                                %Ligadura de frase em quiálteras?		
    c2 (aes g)
    c2 ( ~ \times 2/3 {c4 \< b c \!} \times 2/3 {a'4 \> ees c ) \!}
    b2 (\dimTextDim g \> fis \!)
    r1 s2
    r1 s2
                                %22
    r2 r2 r8 aes, g b
    c d ees c g b c d ees g b c
    <c a'>2:8 <c a'>2:8 <c g'>2:8
    <c fis>2:8 <c fis>2:8  <c f>8 <c f>8 <c f>8 c'
    <c c,>2:8 <c c,>2:8 <bes bes,>2:8	 	  
                                %27
    <c, aes>2:8 <c aes>2:8 <c aes>2:8
    <ces aes>2:8 <ces aes>2:8 <ces aes>2:8
    <ces-. aes>4 r8 g, bes4. ees8 g4. bes,8
    ees4. ges8 bes4. ees,8 ges4. bes8 
    ees4-. \f r4 d4-. r4 r4 r8 bes 
                                %32	
    <ges-. ees'>4 r <f-. d'> r4 r2
    r4 ces'-.  \p (r aes-. r f)
    r4  ges-. (r4 ees-. r4 c-.)
    r8 f f ges aes2 \< ~ (aes8 \> bes \!) bes aes 
    r8 ges ges aes bes2 \< ~ (bes8 \> ges \!) ges f
                                %37	
    r4 ees2 \< (bes'4 ges' bes) 
    aes (ges) f (ges) f (ees) 
    des \f (bes) ges (ees) des' (bes) 
    ges (ees) des' \> (bes) a (ees) \! 
    d \p (d-. r d-. r d-.)	
                                %42
    r ges-. (r ees-. r ees-.)
    r bes'-. (r bes-. r fes-.)
    r ees-. (r aes-. r ees'-.)
    r8 dis  \p dis dis dis cisis cisis dis dis bis bis cis
    r8 c c c c b b c c b b c
                                %47		
    r8 \crescTextCresc cis \< cis cis cis bis bis cis cis2:8 \!
    cis2:8 \< cis2:8 cis2:8 \!
    r4 <d \f a d,>4 r4 fis, (a d)
    cis (<d, a' a'>4) r2 r4 f
    (a d cis a') r4 d, 
                                %52	
    (cis a') r d, (cis a')
    d,,8 \f f f bes <bes d,>1 ~
    <bes d,>2 c,8 ees ees bes' c,8 ees ees aes
    ces, d d aes' bes, ees ees g ces, ees ees aes
    bes, ees ees g c, ees ees ges d f f bes 
                                %57
    bes d d f f2 d,8 f f bes
    bes d d4. (c8) g4. (f8) c'4 ~
    (c8 bes) f4. (ees8) bes'4. (aes8) c4 ~
    (c8 aes) c4. (aes8) f4. (c8) aes4 ~ 
    (aes8 f') ees4. \> (f8) ees4. (f8)  ees4 ~
                                %62		
    (ees8 f) ees4. (f8) \! ees4. \p (f8)  ees4 ~
    (ees8 g) ees4. (f8) ees4. (g8)  ees4 ~		
    (ees8 f) ees4. (f8) ees4. (g8)  ees4 ~		
    (ees8 f) ees4. (g8) ees4. (g8) \< ees4 ~
    (ees8 a) ees4. (g8) d4. (bes'8)  d,4 \!
                                %67 
    d2 (e) \p r2
    r2 r2 c'2 \>
    (bes4) \! r4 r2 r2
    r2 r2 c, 
    (bes) a \< (bes)
                                %72
    c (bes) \! a-. \p
    (r2 bes-.) r2
    d1.
    << {ees1. ~ ees1. ~ ees1. (d)} \\
       {bes1. \p \< (c) \>  (bes) \!  ~ bes1.} >> 
                                %79
    << {ees1. ~ ees1. ~ ees1. (d)} \\
       {bes1. \< (c) \>  (c) \! ~ (b1.)} >> 
    <c ees>1. \p 
                                %ligar semibreve com nota anterior
    a' ~ 
    a ~
    a ~ \key c \major
                                %84
    a8 \p (c a c \repeat "tremolo" 2 {a8 c8} \repeat "tremolo" 2 {a8 c8)} 
    \repeat "tremolo" 2 {a8 (c8} \repeat "tremolo" 2 {a8 c8} \repeat "tremolo" 2 {a8 c8)} 
    \repeat "tremolo" 2 {f,8 (d'8} \repeat "tremolo" 2 {f,8 d'8} \repeat "tremolo" 2 {d,8 bes'8)} 
    \repeat "tremolo" 2 {d8 (gis,8} \repeat "tremolo" 2 {c8 gis8} \repeat "tremolo" 2 {b8 gis8)} 
    \repeat "tremolo" 2 {c8 \< (a8} \repeat "tremolo" 2 {c8 a8} \repeat "tremolo" 2 {c8 a8 \! ) } 
    \crescTextCresc \repeat "tremolo" 2 {c8 \< (a8} \repeat "tremolo" 2 {c8 a8} \repeat "tremolo" 2 {c8 a8)} 

                                %90
    \repeat "tremolo" 2 {d8 (bes8} \repeat "tremolo" 2 {d8 bes8} \repeat "tremolo" 2 {d8 bes8)} 
    \repeat "tremolo" 2 {d8 (b8} \repeat "tremolo" 2 {e8 b8} <d-. e,>2 
    <c-. e,>4 \f r4 c4. a8 f4 r4
    r2 e4. c8 a4 r
    g'4. e8 c2 f 
    <e c'>4. c'8 a2 b 
                                %96
    r4 c4-. \p (r bes-. r g-.)
		r c4-. (r a-. r a-.)
		r8 fis fis  \< g a2 \! \> ~ (a8 fis) \! a fis
		r8 fis fis  \< g a2 \! \> ~ (a8 fis) \! fis  b,
		r2 g'4. \f e8 c4 r4
	     %adicionar o marc.
	%101
		r2 r2 e4. fis8
		g4 r4 c4. b8 a2
		<d, d' b'>4. g'8 e2 fis
		r4 d4-. \p (r f-. r d-.)
		r g4-. (r e-. r e-.) 
		r8 cis cis  \< d e2 \! \> ~ (e8 c) \! e  c
	%107
		r8 cis cis \< d e2 \! \> ~ (e8 c) \! c  fis,
		R1*3/2
		r8 d' d  \< ees f2 \! \> ~ (f8 d) \! d  g,
		R1*3/2
	%111
		r8 dis' dis  \< e fis2 \! \> ~ (fis8 d) \! d gis, \key e \major
		\crescTextCresc <cis, e>1. \p \< 
		cis4. (e8 a4. b8 cis4. e8)
		r8 d \f d a d2 ~ (d8 a) a  a,
	%115		
		gis4. (ais8 bis4. dis8 fis4. gis8)
		r8 a a a, d2 ~ (d8 a') a  a,
		gis4. (ais8 bis4. dis8 fis4. gis8)
		r8 a \f a dis, r8 cis cis cis' cis fis, fis a		
	%119
		r8 d d cis r8 fis, fis fis' fis b, b d
		r8 bis bis cis r8 a a ais r8 bis bis cis
		a fis fis a a e e gis gis fis fis gis,
		fis'4. \f gis8 a2 r2
		b4. cis8 d2 r2
	%124
		dis4. e8 eis4. fis8 fisis4. gis8
		<gis b,>2  ~ <gis b,>8 <a b,>8 <a b,>8 <gis b,>8 <gis b,>2  ~
		<gis b,>8 <a b,>8 <a b,>8 <gis b,>8 <gis b,>2  ~ <gis b,>8 <a b,>8 <a b,>8 <gis b,>8
		<< {\crescTextCresc gis4. \< (a8) a4. (gis8) gis4. (a8)\!} \\ {b,2 b b} >> 
		<< {a'4. \< (gis8) <gis b,>4. (gis,8) <gis' b,>4. \! (<a a,>8)} \\ {b,2 s1} >> 
           %revisar linha acima
	%129
		<a' a,>4-. \f r8 a8-. fis-. e-. e-. r r a, a' a
		<a a,>4-. r8 a8 e d d r r a a' a
		<a a,>4-. r8 a8 d, cis cis r r a a' a		 
		<a a,>4-. r8 a  cis, bis bis r r2 \key ees \major
	%133
		r2 aes,4. \p (c8 c2)
		r2 aes4. (c8 c2)
		r2 aes1 
		(b1 c2) ~
		c2 ~ c4.  \< (aes8 c4. ees8 \!)
		\crescTextCresc ees2 \< r4 r8 g, (c4. ees8)
		ees2 (aes,4) r8 a8 (c4. ees8)
	%140 
		f2 (a,4) r8 bes8 (des4. f8)
		aes4. c,8 ees4. aes8 c4. ees,8
		aes4. c8 ees4. aes,8 c4. ees8
		aes4-. \f r4 <g b, d,>4-. r4 r2
		R1*3/2
		d,4-. \p r4 cis-. r4 r2
		R1*3/2		
		cis4-. \pizz r4 b-. r4 r2
	    	R1*3/2
	%149
		r4 a r g r d'
		r4 d r c \> r a
		g'4. \! \arco (aes8 g1) ~
	  	g4. (b8 c2 g) ~
		g4. (aes8 g2. c4)
		b4. (c8 d2 b)
		c2 (aes g)
            %adicionar espress. 
	%156
		c ~ (\times 2/3 {c4 \< b c \! } \times 2/3 {f4 \> c aes \!)} 
		c2 (aes g)
	 	bes ~ (\times 2/3 {bes4 \< a bes \! } \times 2/3 {g'4 \> e bes \!)}
		a2 \> (f e) \!
		R1*3/2
		R1*3/2
		R1*3/2
		c2 \p (\crescTextCresc bes \< g)
	%164
		aes8 \< g aes b c e f g aes b c f
		<f aes>2:8 \f <f aes>2:8  <f aes>2:8
		<f aes>2:8  <f aes>2:8 \crescTextCresc <f aes>2:8 \< 
		<f ges>2:8  <f ges>2:8  <ees ges>2:8
		<f aes,>2:8  <f aes,>2:8 <f aes,>2:8	
	%169
		<fes aes,>2:8  <fes aes,>2:8 <fes aes,>2:8
		<fes aes,>4-. r8 c,8 ees4. aes8 c4. ees,8	
		aes,4. c8 ees4. aes,8 c4. ees8
		<ees c'>4-. r4 <d b'>4-. r4 r4 r8 ees8
		<ees c'>4-. r4 <d b'>4-. r4 r2 
		r4 aes'4-. \p (r4 f-. r4 d-.)
	%175
		r4 ees4-. (r4 c'-. r4 a-.)
		r8 d, d \< ees f2 \! \> ~ (f8 g \!) g f
		r8 ees ees \< f g2 \! \> ~ (g8 ees \!) ees d
		r4 c2 (g'4 \< ees' g)
		f ( ees d ees d c) 
	%180  
		bes \f ( g ees c) bes' ( g 
		ees c) bes' \> ( g fis c)
		b \p b-. (r4 b-. r4 b-.)
		r4 ees-. (r4 c-. r c-.)
		r4 g'-. (r4 g-. r4 des-.)
	%185
		r4 c-. (r4 f-. r4 c'-.)
		r8 c \p c c c b b c c a a b 
		r8 a a a a gis gis a a gis gis a
		r8 \crescTextCresc bes \< bes bes bes a a bes bes2:8
		bes2:8 \<  bes2:8  bes2:8 \!
	%190
		r4 <b fis b,>4 \f r4 dis,4 (fis b)
		ais ( < d, b' fis' >4 ) r2 r4 d 
		(fis b, ais fis') r4 b,
		(ais \< fis') r4 b (ais fis') \!
		b,,8 \f d d g <g b,>1 ~
	%195
		<g b,>2 a,8 c c g' a, c c f
		aes, b b f' g, c c e aes, c c f
		g, c c e a, c c ees b d d g
		g b b d d2 r4 b,8 d
		d g g b c4.-> ( a8) f4.-> ( d8)
	%200
		b'4.-> (g8) e4.-> (c8) a'4.-> (f8)
		a'4.-> (f8) d4.-> (a8) f4.-> (d8)
		<d f>4 \dimTextDim <d f>4-. \> ( r4 <d f>4-. r4 <d f>4 )
		r4 \> <d fis>4 r4 <d fis>4 r4 <d a'>4 \p
	    %adicionar o p dolce
	    	r4 <g e>4 r4 <fis d>4 r4  <g e>4
	%205
		r4 <a d,>4 r4 <fis d>4 r4 <g e>4
		r4 <fis d>4 r4 <g e>4 r4 fis
		r4 a4 \< r4 g r g 
		f2-. \p (f2-.) r2
		r2 r2 a2 
	%210
		(g4) r4 r2 r2 
		r2 r2 <c c,>2 ~
		<c c,>2 <c c,>1 \<
		<c c,>1 <c e,>2-. \p 
		_(r2 <e, c>2-.) r2	
	%215
		<g b,>1. \p
		<< { << g1. ({ s2 s4 \< s4 s4 s4 \!} >> a1.) (g)  ~ g } \\ {c,1. ~ << c1. ~ { s4 \> s4  s4\!} >> c1. (b) } >>
		<< c1. ~ { s4 \p s4 s4 \< s4 s4 s4 \!} >> << c1. ~ { s4 \> s4  s4\! s4} >> c1. (b1.) \time 2/2	
	    %adicionar o piú 
	%224
		r4 \p \crescTextCresc bes4-. \< ( r g-.)
	    %adicionar o crescendo ed agitato	
		r4 c4-. ( r c-.)
		r8 d d ees f aes, aes'4 ~
		aes8 \< g g f f <d b'>8 <d b'>8 <d b'>8 \!
		r4 <e g,>4-. \p (\crescTextCresc r4 \< <g g,>4-.)
		r4 <aes c,>4-. (r4 <aes f>4-.) 
		r8 d d ees f aes, aes'4 ~
	%231
		aes8 \< g g f f <d, b'>8 <d b'>8 <d b'>8 
		<e c'>1:8 \f 
		c4. d8 e g c e
		<f a,>1:8 
		f,4. g8 a c f a 
		<f aes>1:8 \f
		<f des>1:8	
	%238
		<ees c>1:8
		<d b>1:8
		<f aes,>1:8
		<aes, f>1:8
		<g ees>1:8
		<d b>1:8
		c2:8 bes2 \fz ~ 
		bes8 bes8-. bes8-. bes8-. a2 \fz ~
	%246
		a8 a-. a-. a-. aes2 \fz ~
		aes8 aes8-. aes8-. aes8-. des2:8
		c2:8 bes2 \fz ~
		bes8 bes8-. bes8-. bes8-. a2 \fz ~
		a8 a-. a-. a-. aes2 \fz ~
		aes8 aes8-. aes8-. aes8-. des2:8
	%252
		<c e>1:8
		\dimTextDim <c e>1:8 \>
		\times 2/3 { <c e>4-. <c e>4-. <c e>4-. } \times 2/3 { <c e>4-.
<c e>4-. <c e>4-. }
		\times 2/3 { <c e>4-. <c e>4-. <c e>4-. } \times 2/3 { <c e>4-. 
<c e>4-. <c e>4-. }
		c4-. c4-. c4-. c4-.
		c4-. c4-. c4-. c4-.
		c4-. r4 r2
		c4-. r4 r2
		<< <g' e'>1 ^\fermata { s4 \p s4\< s4\! \> s4\! } >> 
		

 }
}

viola= { 
	\clef alto 		
	\relative c {
	 	
		ees8 \p ees8 ees ees ees2:8 ees 2:8
		ees2:8 ees2:8 \crescTextCresc ees2:8 \< 
		ees2:8 ees2:8 ees2:8
		ees2:8 ees2:8 des2:8
		c2:8 c2:8 \< c2:8
		<c c'>2:8 <c c'>2:8 <c c'>2:8 \!
	%7
		
		<c'-. ees>4 \f r <g g'>1 \fp ~ 
		<g g'>1. 
		des'4-. \p r <f f,>1 ~ 
		<f f,>1. 
		f,4. \p (g8 aes2 f4) r
		bes4. (des8 \< f2 \> f,4) \! r
		f4. (g8 aes2 f4) r
		b4. \< (d8  f2 \> f,4) \! r
	%15
		g4. (b8 c2) r2
		r2 aes (c)
		g4. (b8 c2) r2
		r2 a (c)		
		g4 r4 r2 r2
		\dimTextDim b \> (g fis)
		r2 fis r2
	%22
		g \< (ees d) \!
		c4. \< (d8 ees4. g8 c4. ees8) \!
 		ees2 \f (fis,4) r8 g (c4. ees8)
		ees2 (aes,4) r8 \crescTextCresc a \< (c4. ees8)
		f2 (a,4) r8 bes (des4. f8)
	%27		
		aes4. c8-. \sf aes4. ees8-. c4. aes'8-. \sf 
		ees4. ces8-. aes4. ees'8-. \sf  ces4. aes8-.
		ees4-. r <bes' g'>2:8 <bes g'>2:8 
		<bes ges'>2:8 <bes ges'>2:8 <bes ees,>2:8 
		<a-. ges'>4 \f r <bes-. f'>4 r r2
	%32
		<a-. a'>4 r  bes8-. \fp bes'-. bes-. bes,8-. bes8-. bes'-. bes-. bes,8-.
		bes8 aes' aes bes,8 bes8 f' f bes,8 bes8 d d bes8 
		bes8 ees ees bes8 bes8 ges' ges bes,8 bes8 a' a bes,8 
		bes8 bes' bes bes,8 bes8 bes' bes bes,8 bes8 bes' bes bes,8
		bes8 bes' bes bes,8 bes8 bes' bes bes,8 bes8 bes' bes bes,8
	%37
		ges' bes bes ges ees \< ges es ees ges des des ges
		aes,8 aes' aes aes,8 aes8 aes' aes aes,8 aes8 aes' aes aes,8 \!
		ges'8 ges, ges ges'8 ges8 ges, ges ges'8 ges8 ges, ges ges'8
		ges8 ges, ges ges'8 f8 \> f, f f'8 f8 f, f f'8 \!
		bes,4 \p f'-. r f-. r ces'-.
	%42
		r bes-. (r bes-. r ges-.)
		r g-. (r g-. r bes-.)
		r8 g,-. g-. ces-. ces-. aes-. aes-. ees'-. ees-. ces ces aes' 
		c,1 \p cis 2
		d1 dis 2  
	%47		
		\crescTextCresc e \< eis fis
		g \< e g, \!
		r4 fis \f (a d cis fis)
		e (a,) r f (a d
	%51 e 52	
		cis ) << {a'2 a a4 ~ a a2 a2 s4  } \\ {f e d cis f (e) d (cis) f (e) a,} >>  
	%53   
	        d,8 \f f f bes <bes f'>1 ~
	    	<bes f'>2  ees4 (c) ees4 (c)	
		d4 (ces) ees4 (bes) ees4 (ces)
		ees4. (bes8)  a c c ees, bes' d d  f 
	%57
		f bes bes d d2. \clef treble d,8 f
		f bes bes d ees4.-> (c8) aes4.-> (f8)
		d'4.-> (bes8) g4.-> (ees8) c'4.-> (aes8) \clef alto
		c4.-> (aes8) f4. (c8)  aes4. (f8)
		<f aes>4 \dimTextDim <f-. aes>4 \> (r <f-. aes>4 r <f-. aes>4)  
	%62
		r <f a>4 r <f a>4 \! r <f c'>4 \p	  
		r <g bes>4 r <f a>4 r <g bes>4
		r <f c'>4 r  <f a>4  r <g bes>4 
		r <f a>4 r <g bes>4 r a	\<	
		r c r bes r bes \!
	%67
		<aes-. aes'>2  (<aes-. aes'>2) \p  r2
		R1*3/2
		<bes-. f'>2  (<a-. c>2) r2
		r2 r2 ees' ~ 		
		ees ees,1 \< 
	%72
		ees \! \< ees2-. \p 
		(r2 f-.) r2
		aes1.
		g2 \p \<(ges1) \! ~
		ges1. \> 
		f1. \! ~
		f2 (bes aes)
	%79
		g (fis1 \<
		a1.) \! 
		(g1. ~ g1.)
 		a1. \p
	     %ligar semibreve com nota anterior
		<< {c1. _~ c1. } \\ 
		{ees1. ^(e1.)
		 ^~ e1. } >>  \key c \major
	%84	
		a,4. \p (b8 \< c4. e8 a4. c8 \!)
		c2 \> (ees,4 \!) r r2
		bes4. \p (c8 \< d4. f8 bes4. d8 \!)
		d2 \> (gis,4 \!) r r2 
		\repeat "tremolo" 2 {a,8 \< (c8} \repeat "tremolo" 2 {a8 c8} \repeat "tremolo" 2 {a8 c8 \!)} 
		\crescTextCresc \repeat "tremolo" 2 {f,8 \< (c'8} \repeat "tremolo" 2 {f,8 c'8} \repeat "tremolo" 2 {f,8 c'8 )} 
	%90
		\repeat "tremolo" 2 {f8 (d8} \repeat "tremolo" 2 {f8 d8} \repeat "tremolo" 2 {f8 d8)} 	
		\repeat "tremolo" 2 {b'8 (gis8} \repeat "tremolo" 2 {b8 e,8)} e2-.
		a,4. \f b8 c4 r a'4. c8
	    %colocar o marc.
		c4 r c,4. a8 f4 r
		c'4. c8  a2 b
		c4 r r2 r2
	%96
		c,8-. \p c'-. c-. c,-. c8-. c'-. c-. c,-. c8-. c'-. c-. c,-. 
	        c8 c' c c,  c8 c' c c,    c8 c' c c,
		b' c c b b c c b b dis dis c
		b c c b b e e b b dis dis b 
		r2 r e4. \f gis8
	%101
		g4 r b,4. g8 e4 r
		<d g d'>4. b'8 g2 c
		b4 r r2 r2 
		g8 \p g' g g, g g' g g, g g' g g, 
		g g' g g, g g' g g, g g' g g, 
		fis fis' fis fis, fis fis' fis fis, fis fis' fis fis,
	%107
		fis fis' fis fis, fis fis' fis fis, fis fis' fis fis,
		fis fis' fis fis, fis fis' fis fis, fis fis' fis g
		g, g' g g, g g' g g, g g' g g, 	
  		g g' g g, g g' g g, g g' g gis 	
	%111		
		gis, gis' gis gis, gis gis' gis gis, gis gis' gis gis, \key e \major
		gis \p \crescTextCresc gis' \< gis gis, gis gis' gis gis, gis gis' gis gis,
		a e' e e, e e' e e, e e' e a,
		a \f a' a a, a a' a fis fis d' d f, 
	%115
		fis gis gis fis fis gis gis bis, bis dis dis gis, 
		a a' a a, a a' a fis fis dis' dis fis, 
		fis gis gis fis fis gis gis bis, bis dis dis gis, 
		fis4. \f gis8 a2 r2
	%119
		b4. cis8 d2 r2
		fis4. e8 dis4. cis8 fis4. e8
		a4-. r4 cis,-. r4 gis4-. r4
		r8 cis-. \f cis bis r gis' gis fis fis d d cis
		r  fis fis eis e cis' cis b b g g fis
	%124
		r a a gis r b b a r cis cis b
		r d d cis cis2 ~ cis8 d d cis 
		cis2 ~ cis8 ~ d d cis cis2 ~ 
		\crescTextCresc cis8 \< (d) (d4 ~ d8 e,) (e4 ~ e8 d) d4 ~
		(d8 e,) e4 ~ (e8 d) d4 ~ (d8 e e4)
	%129
		< cis a'>4 \f r8 a''8-. fis-. e-. a,-. fis-. fis-. e-. cis-. cis-.  
	  %colocar o marc. acima
		fis4-. r8 a'8 e d a e e d fis fis
		e4-. r8 a'8 d, cis a d, d cis e e
		dis4-. r8 a''8 cis, bis a cis, cis bis cis cis \key ees \major
	%133
		c2:8 \p c2:8 c2:8
		c2:8 c2:8 c2:8
		c2:8 c2:8 c2:8
		c2:8 c2:8 c2:8
		c2:8 c2:8 \< <ees c'>2:8
		\crescTextCresc <ees c'>2:8 \<  <ees c'>2:8  <ees c'>2:8
		 <ees c'>2:8  <ees c'>2:8  <ees c'>2:8  
	%140
		<f c'>2:8  <f c'>2:8  <f bes>2:8
		<ees c'>2:8  <aes c>2:8  <aes c>2:8  
		<aes c>2:8 <aes c>2:8 <aes c>2:8 
		<ees c'>4-. \f r4 <g g'>1 \fp ~ 
		<g g'>1.
		d'4-. \p r4 <gis gis,>1 ~ 
		<gis gis,>1.
		e4-. r4 d1 ~ 
		d1.
	%149
		d2 (cis c
		bes aes fis)
		g4. (aes8 g1) ~
	    %adicionar arco.
		g4. (b8 c2 g) ~
		g4. (aes8 g1) ~
		g4. (a8 b2 g)
		g4. (b8 c1) ~
	%156
		c4 (g' aes f c f,)
		g4. (b8 c2) r2
		\dimTextDim  c4. \> (d8 e2 c)
		c,2. \> (d4 e) \! r4
		a2 \> (f e) \!
		r2 e2-_ r2
	     %colocar tenuto mais staccato
	 	g2 \p (\crescTextCresc e \< des)	
		<c c'>1. 
	%164
		f4. \< (g8 aes4. c8 f4. aes8) \!
		aes2 \f (b,4) r8 c8 (f4. aes8)
		aes2 (des,4) r8 d8 (\crescTextCresc f4. \< aes8)
		bes2 (d,4) r8 e8 (ges4. aes8)
		des4. f8-. \sf des4. aes8 f4. des'8-. \sf 
	%169
		aes4. fes8 des4. a'8-. \sf fes4. des8
		aes4-. r4 <ees c'>2:8  <ees c'>2:8
		<ees c'>2:8  <ees c'>2:8 <ees c'>2:8
		<ees c'>4-. r4 <aes g'>4-. r4 r4 r8 c8
		<c aes'>4-. r4 g8-. \fp g'-. g-. g,-. g-. g'-. g-. g,-.
		g g'g g,  g g'g g, g g'g g,
	%175
		g g'g g,  g g'g g, g g'g g,
		g g'g g,  g g'g g, g g'g g,
		g g'g g,  g g'g g, g g'g g,	
		ees' g g ees c ees ees c ees bes bes ees
		f, f'f f,  f f'f f, f f'f f,
	%180
		ees' \f ees, ees ees' ees ees, ees ees' ees ees, ees ees' 	
		ees ees, ees ees' d \>  d, d d' d d, d d'
		g,4 \p d'-. (r4 d-. r4 aes'-.)
		r4 (g-. r g-. r ees-.)
		r4 (e-. r e-. r g-.)		
	%185
		r8 e,-. e-. aes aes f f c' c aes aes f'
		a,1 \p  bes2 
		b1 c2
		\crescTextCresc des2 \< d e
		e \< cis e, \!
	%190
		r4 dis \f (gis b ais dis)
		cis ( fis) r4 d, (fis b 
		ais) d ( cis) b ( ais) d
		(cis) \< b (ais) d ( cis fis,)
		b8 \f ees, ees g <g d'>1 ~
	%195	
		<g d'>2 c4 (a) c (a)
		b (aes) c ( g) c (aes)
		c4. (g8) fis a a c, g' b b d 
		d g g b b2 b,8 d d g
		g b b4. (a8) e4. (d8) f4 ~
	%200
		(f8 d) d4. (c8) e4. (c8) a'4 ~
		(a8 f) a4. (f8) d4. (a8) f'4 ~
		(f8 d)\dimTextDim c4. \> (d8) c4. \> (d8) c4 ~
		(c8 d) c4. (d8) c4. (d8) c4 \p ~ 
	    %colocar p dolce
		(c8 e) c4. (d8) c4. (e8) c4  ~
	%205
		(c8 d) c4. (d8) c4. (e8) c4  ~
		(c8 d) c4. (e8) c4. (e8) c4  ~
		(c8 fis) c4. \< (e8) b4. (g'8) b,4 
		b2-. \p (c-.) r2 
		r2 r2 f2
	%210
		( <d g,>2-.)  (< c a >2-.) r2 
		r2 r2 a 
		(g) fis \< (g)
		a (g) fis-. \p 
		(r2 g-.) r2
	%215
		f'1. \p 
		e2 (ees1) ~ \<
		<< ees1. ({ s4 \> s4  s4\! s4} >>
		e1.)
		(d2) (f,1)
		<< {g2 \p ~ g1 \< } \\ \stemUp {e2 (ees1)} >>
	  % acima colocar o piú p
	  	<< { << a1. ({ s4 \> s4  s4\! s4} >> g1.) ~ g} \\ {ees1. ~ ees (d)} >>  \time 2/2
	 %224
		r4 \p \crescTextCresc e4-. \< (r4 e4-.)
		r4 f4-. (r4 aes4-.)
		r4 aes'-> ~ aes8 f f c
		b4-. \< aes2-> g4-.
		c,8 \p \crescTextCresc c' \< c c, c c' c c,
		c c' c c,  c c' c c,  
		c c' c c,  c c' c c,  
	%231
		c \< c' c c,  c c' c c,
		c4. \f d8 e g c e 
		<e g,>1:8
		f,4. g8 a c f a
		<f a>1:8
		<f aes>1:8 \f
		<f aes>1:8
	%238
		<ees c'>1:8
		<d f>1:8
		<c aes'>1:8
		<des f>1:8
		<g, ees'>2:8 c2:8
		g1:8 
		<c, c'>2:8 <c c'>2 \fz ~ 
		<c c'>8 <c c'>8-. <c c'>8-. <c c'>8-. <c c'>2 \fz ~
	%246
		<c c'>8 <c c'>8-. <c c'>8-. <c c'>8-. <des' f>2 \fz ~
		<des f>8 <des f>8-. <des f>8-. <des f>8-. <f, aes>2:8
		<c c'>2:8 <c c'>2 \fz ~ 
		<c c'>8 <c c'>8-. <c c'>8-. <c c'>8-. <c c'>2 \fz ~
		<c c'>8 <c c'>8-. <c c'>8-. <c c'>8-. <des' f>2 \fz ~
		<des f>8 <des f>8-. <des f>8-. <des f>8-. <f, aes>2:8
	%252
		<g g'>1:8 
		\dimTextDim <g g'>1:8 \>
		\times 2/3 {g4-. g-. g-.} \times 2/3 {g4-. g-. g-.}
		\times 2/3 {g4-. g-. g-.} \times 2/3 {g4-. g-. g-.}
		e4-. e4-. e4-. e4-.
		e4-. e4-. e4-. e4-.
		e4-. r4 r2
		e4-. r4 r2
		<< <e c'>1 ^\fermata { s4 \p s4\< s4\! \> s4\! } >>
		
	 

  }
}

violoncelo= { 
	\clef bass 		
	\relative c, {
	 	
		
	c8 \p c' c  c c2:8 c2:8
		c2:8 c2:8 \crescTextCresc c2:8 \< 
		c2:8 c2:8 c2:8 
		c2:8 c2:8 bes2:8
		aes2:8 aes2:8 \< aes2:8 		
		ees2:8 ees2:8 ees2:8 \! 
	%7
		c4-. \f r g'4-. \f r  r2
		R1*3/2
		bes4-. \p r f'4-. r r2 
		R1*3/2
		aes,1. \p
		bes1.
		aes1.
		g1.
	%15
		g2 ~ (g4. b8 c2)
		aes4. (c8 f2 aes,4) r
		g2 ~ (g4. b8 c2)
		fis,4. (b8 c2 fis,4) r
		g4. (a8 \dimTextDim b2 \> ~ b4. cis8)
		d2 \> ~ d4. e8 \! fis2
		r2 fis r2		
	%22
		g, \< (ees d) \!
		c4. \< (d8 ees4. g8 c4. ees8) \!
 		ees2 \f (fis,4) r8 g (c4. ees8)
		ees2 (aes,4) r8 \crescTextCresc a  \< (c4. ees8)
		f2 (a,4) r8 bes (des4. f8)
	%27		
		aes4. c8-. \sf aes4. ees8-. c4. aes'8-. \sf 
		ees4. ces8-. aes4. ees'8-. \sf ces4. aes8-.
		ees4-. r ees'1->
		ges,1.->
		ees4-. \f r4 bes'4-. r4 r2
	%32
		ees,4-. r4 bes'4-. r4 r2
		R1*3/2
		R1*3/2 
		\clef tenor bes''1 \p (d,2) 
		bes'2 (c, d) \clef bass
	%37
		ees ges, \< bes 
		des, f aes, \!
		bes \f ces des
		ees f \> f, \!
		bes4 \p (ces'-. r aes-. r f-.)	
	%42
		r bes-. (r ges-. r ees-.)
		r8 des-. des-. g-. g-. fes-. fes-. bes-. bes-. g g des' 
		des4 (ces-.) (r ees-. r ees-.) \clef tenor
		r8 fis \p  fis gis a2 ~ (a8 gis) gis a
		r8 fis fis gis a2 ~ (a8 gis) gis a 
	%47		
		r8 \crescTextCresc a \< a a a gis gis a a2:8
		a2:8 \< a2:8 a2:8 \!
		R1*3/2 \clef bass
		r4 f,,4 (a d cis f)
		e (a,) r f' \< (e a,)
	%52	       
		r f' (e a,) \! r2
		aes \f aes'1
		g2 g, f
		f' g aes 
		bes bes, bes'
	%57
		aes aes,2.. aes'8
		g4. g,8 c,4. c'8 f,4. f'8
	 	bes,4. bes8 ees,4. ees'8 aes,4. aes8
		f4 r8 aes' f4. (c8) aes4. (f8) 
		ces'4. ( \dimTextDim f,8) \> ces'4. (f,8) ces'4. (f,8)	
	%62
		c'4. (f,8) \! c'4. (f,8) a4. \p (f8)
		bes4. (f8) c'4. (f,8) bes4. (ees,8)
		a4. (f8) c'4. (f,8) bes4. (ees,8)
		c'4. (f,8) bes4. (ees,8) c'4. \< (c,8)
		ees'4. (d,8) d'4. (g,8) g'4. \! (bes,8)			
	%67
		bes'2-. (c,-.) \p r2
		R1*3/2
		d2-. (ees-.) r2
		R1*3/2
		r2  c2-. \< (bes-.) \!
	%72
		a \< (bes) \! c-. \p
		(r2 bes-.) r2 
		bes1. 
		ees,4. \p  \< (f8 ges4. bes8 ees4. ges8) \!
		ges2 \> (a,4) \! r4 r2
	        bes2 \pizz r2 r2		 
	        bes2 r2 r2
	%79
		ees,4. \arco \< (f8 ges4. bes8 ees4. fis8) \!
	     	fis2 \> (fis,4) \! r4 r2
	        g2 \pizz r2 r2		 
	        g2 r2 r2
		fis'2 \p (fis,4) r4 r2
 		f'2 (f,4) r4 r2
		e'1 (c2 ~
		c2) aes1 \key c \major
	%84
		e1. \p 
		f
		d
		e
		e \< 
		\crescTextCresc f4. \< (g8 a4. c8 f4. a8)
	%90
		d,,1.
		e4. fis8 gis4. b8 e4 r
		a,4-. \f r r2 r2
		c,1 d2
            %colocar o marc.
		e4 r f'4. e8 d2
		c4 r f'4. e8 d2
	%96		
		r2  c,,4 \pizz \p r4 e r
	   	r2 f4 r4 a4 r4
		b4 r4 r2 r2
		b4 r4 r2 r2 \arco
		e4. \f fis8 g4 r4 r2
	    %colocar o marc.
	%101
		e4. fis8 g4. e8 c4 r4
		b4. g8 e2 fis
		g4 r4 c'4. b8 a2
		r2  g,4 \pizz \p r4 b r
		r2  c4 r4 e r
		fis4 r4 r2 ais4 r4
	%107 
		fis4 r4 r2 r2 
		d,4. \arco \p \< (fis8 b4. d8 fis4. g8 \!)
		g2 \> (b,4 \!) r4 r2
		ees,4. \< (g8 c4. ees8 g4. gis8 \!)
	%111
		gis2 \> (bis,4 \!) r4 r2  \key e \major
		e,4. \p ( \crescTextCresc fis8 \< gis4. cis8 e4. gis8)
		cis,,1.
		d4. \f (e8 fis4. a8 d4. fis8)
	%115
		gis,1.	
		d4. (e8 fis4. a8 d4. fis8)
		gis,1.	
		fis4. \f gis8 a2 r2
	%119
		b4. cis8 d2 r2
		dis4. e8 fis4. fisis8 gis4. a8
		fis4-. r4 gis r4 gis,-. r4
		r8 a'-. \f a-. gis-. r8 cis cis fis, fis b, b fis
		r8 d'' d cis r8 fis fis b, b e, e b \clef tenor
	%124
		r8 bis' bis cis r8 cis cis d r8 dis dis e
		r8 e e eis eis2 ~ eis8 e e f 
		f2 ~ f8 e e f f2 ~
		\crescTextCresc f8 \< (e) e4 ~ (e8 d) d4 ~ (d8 e,) \clef bass e4 ~
		(e8 d) d4 ~(d8 e,) e4 ~ (e8 d) d4
	%129
		cis4-. \f r r2 r8 a' cis, cis 
	   %adicionar o marc.
		fis4-. r4 r2 r8 a fis fis
		e4-. r4 r2 r8 a e e
		dis4-. r r2 r8 a' cis, cis  \key ees \major
	%133
		c4. \p (ees8 aes1)
		c,4. (ees8 aes1)
		c,1 (ees2 ~
		ees aes1)
		aes2:8 aes2:8 \< aes2:8
		\crescTextCresc a2:8 \< a2:8 g2:8
		fis2:8 fis2:8 f2:8
	%140
		ees2:8 ees2:8 des2:8
		c2 aes''-> ees-> 
		c-> aes-> ees->
		c4-. \f r4 g'-. r4 r2
		R1*3/2
		b4-. \p r4 fis'-. r4 r2 
		R1*3/2 
		<fis ais,>4-. \pizz r4 <fis b,>4-. r4 r2
		R1*3/2
	%149
		d,4 r e r fis r
		g r c r d r
		b1 \arco \p ~ (b4 f')
		ees1 ~ (ees4 c)
		b1 ~ (b4 a)
		g1. 
		g2 ~ (g4. b8 c2)
	%156
		aes4. (c8 f2 aes,)
		g2 ~ (g4. b8 c2) ~
		c2 (\dimTextDim c,1 \>) ~
		c1. \> ~ 
		c2. (d4 e2) \!
		r2 e-_ r2
	   %colocar o tenuto
	   	R1*3/2
		c'2 \p (\crescTextCresc bes2 \< g)
	%164
		f4. \< (g8 aes4. c8 f4. aes8) \!
		aes2 \f (b,4) r8 c8 (f4. aes8)
		aes2 (des,4) r8 d8 (\crescTextCresc f4. \< aes8)
		bes2 (d,4) r8 ees8 (ges4. bes8)
		des4. f8-. \sf des4. aes8 f4. des'8-. \sf 
	%169
		aes4. fes8 des4. a'8-. \sf fes4. des8
		aes4-. r4 aes'2:8  aes2:8
		c,2:8 c2:8 c2:8
		c,4-. r g'-. r r2 
		c,4-. r g'-. r r2 \clef tenor
		f''2 \p (d b)
	    %colocar o dolce
	%175
		c (ees fis)
		g1 (b,2)
		g'2 (a, b) \clef bass
		c ees, g
		bes, d f,
	%180
		g \f aes bes c d \> d, \!
		g4 \p (aes'-.) (r f-. r d-.)
		r g-. (r ees-. r c-.)
		r8 bes bes e e des des g g e e bes'
	%185
		bes4 (aes-.) (r c-. r c-.) \clef tenor
		r8 ees ees f ges2 ~ (ges8 f) f ges
		r8 ees ees f ges2 ~ (ges8 f) f ges
		r8 \crescTextCresc ges \< ges ges ges f f ges ges2:8
		ges2:8 \< ges2:8 ges2:8 \!
	%190
		R1*3/2 \clef bass
		r4 d,, \f (fis b ais d)
		cis (fis,) r4 d' (cis fis,)
		r4 d' \< (cis fis,) \! r2 
		f2 \f f'1
	%195
		e2 e, d 
		d' e f
		g g, g'
		f f,2. f'4
		e2 a,4. a'8 d,,4. d'8
	%200
		g,4. g'8 c,,4. c'8 f,4. f'8
		d,4 r8 f'8 d4. (a8) f4. (d8)
		aes'4. (\dimTextDim d,8 \>) aes'4. (d,8) \> aes'4. (d,8)   
		a'4. (d,8) a'4. (d,8) fis4. (d8) \p
	    %adicionar acima p dolce
	    	g4. (c,8) a'4. (d,8) g4. (c,8)
	%205   
		fis4. (d8) a'4. (d,8) g4. (c,8)
		a'4. (d,8) g4. (c,8) a'4. (a'8)
		<< c,4. ({ s4 \< s8 } >> b8) b4. (e8) e4. (g8)
		g,2-. \p (a2-.) r2 
		R1*3/2 
	%210
		b2-. (c2-.) r2 
		R1*3/2
		r2 a2  \< (g)
		fis (g) a-. \p 
		(r2 g2-.) r2
	%215
		g \pizz r2 r2 
		c,4. \arco (d8 \< ees4. g8 c4. ees8)
		ees2 \> (fis,4) \! r4 r2
		g2 \pizz r2 r2
		g2 r2 r2
		c,4. \arco  \p (d8 \< ees4. g8 c4. ees8)
	   %adicionar o piú p
		ees2 \> (fis,4) \! r4 r2
		g2 \pizz r2 r2
		g2 r2 r2 \time 2/2
	%224
		c,8 \p \crescTextCresc c' \< c c, c c' c c,  
	     %adicionar acima o crescendo ed agitato
	     	c c' c c,  c c' c c, 
		c c' c c,  c c' c c,
		c \< c' c c,  c c' c c, \!
		r4 bes'4-. \p (\crescTextCresc r4 \< e,4-.)
		r4 f-. (r aes-.) \clef tenor 
		r4 aes''-> ~ aes8 f f c
	%231
		b4-. \< aes2-> g4 \clef bass 
		c,,4. \f d8 e g c e
		e2 c
		f,4. g8 a c f a
		c2 c,
		f,4. \f g8 aes c f aes
		aes,4. c8 des f aes des
	%238
		g,,4-. g'' (aes g
		f d b g)
		f,4. g8 aes c f aes
		aes,4. c8 des f aes des
		g,,4-. g' (aes g
		f d b g)
		c,4. d8 e-. g-. c-. e-. 
		c,4. e8 f a c f
	%246
		c,4. e8 f aes des f 
		aes2 f
		c,4. d8 e g c e 
		c,4. e8 f a c f
		c,4. e8 f aes des f 
		aes2 f
	%252
		c,4. d8 e g c e 
		\times 2/3 {\dimTextDim c,2 \> d4} \times 2/3 {e4 g c}
	    	\times 2/3 {c,2 d4} \times 2/3 {e2 g4}
		\times 2/3 {c,2 d4} \times 2/3 {e2 g4}
		c,2 g' ~ 
		g c, ~
		c g' ~
		g c, ~
		<< c1 ^\fermata { s4 \p s4\< s4\! \> s4\! } >>
		
		
		 

  }
}

%% score

\book {
  \header {
    title = "Quarteto n.1 em Dó menor"
    composer = "Johannes Brahms"
    opus = "Opus 51"
    copyright = "© Genos"
  }
  \score {
    \new StaffGroup <<
      \new Staff <<
        \set Staff.midiInstrument = "violin"
        \global
        \violinoum
      >>
      \new Staff <<
        \set Staff.midiInstrument = "violin"
        \global
        \violinodois
      >>
      \new Staff <<
        \set Staff.midiInstrument = "viola"
        \global
        \viola
      >>
      \new Staff <<
        \set Staff.midiInstrument = "cello"
        \global
        \violoncelo
      >>
    >>
    \midi { }
    \layout { }
  }
}
