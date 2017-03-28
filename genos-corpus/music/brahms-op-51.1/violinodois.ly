\version "2.12.0"

pizz = ^\markup{\italic{pizz.}}
arco = ^\markup{\italic{arco}}

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
