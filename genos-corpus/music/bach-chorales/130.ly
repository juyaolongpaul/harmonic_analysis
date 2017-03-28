
\version "2.10.33"

\header {
  title = "130 - Meine Seel erhebt den Herren"
  composer = "J. S. Bach"
}

global =  {
  \time 4/4 
  \key e \minor
}

soprano = \relative c {
  b''2 d 
  b4 b b b 
  c2 b 
  a a 
  g1 
  b2 d 
  \time 4/1
  a\breve a\breve
  \time 2/4
  e2
  \time 4/4
  g fis
  e1 
}


alto = \relative c {
  e'2 d 
  d4 dis e fis 
  e d2 e4 
  e2 d 
  d1 
  d 
  d\breve d\breve
  e2
  e dis
  b1 
}


tenor = \relative c {
  g'2 a 
  b4 a g fis 
  g a2 g4 
  g2 fis 
  b1 
  g
  fis\breve fis\breve
  a2 b2. a4 g1 
}


baixo = \relative c {
  e2 fis 
  g4 fis e dis 
  e fis g e 
  c a d2 
  g,1 
  g'2 b, 
  d\breve d\breve
  c2 b4 a 
  b2 e,1 
}


\score {
  <<
    \new StaffGroup <<
      \override StaffGroup.SystemStartBracket #'style = #'line 
      \new Staff {
        <<
          \global
          \new Voice = "soprano" { \voiceOne \soprano }
          \new Voice = "alto" { \voiceTwo \alto }
        >>
      }
      \new Staff {
        <<
          \global
          \clef "bass"
          \new Voice = "tenor" {\voiceOne \tenor }
          \new Voice = "baixo" { \voiceTwo \baixo \bar "|."}
        >>
      }
    >>
  >>
  \layout {}
  \midi {}
}