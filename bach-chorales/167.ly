\version "2.10.33"

\header {
  title = "167 - Du großer Schmerzensmann"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key e \minor
}


soprano = \relative c'' {
  \partial 4 g4 
    g4 g4 g4 a4 
  b2 r4 b4 
  b4 d4 g,8 a8 b4 
  a2 g4 b4 
  a4 b4 g4 a4   %5
  fis2 r4 g4 
  g4 g4 a4 g4 
  fis2 e4 fis4 
  fis4 fis4 fis4 g4 
  a2 r4 a4   %10
  a4 b4 c4 c4 
  b2 r4 b4 
  b4 d4 c4 b4 
  a2 r4 a4 
  b4 a4 g4 fis4   %15
  e2.
  
}

alto = \relative c' {
  \partial 4 e4 
    d4 e4 d4 e8 fis8 
  g2 r4 g4 
  g4 g8 fis8 e8 fis8 g4 
  g4 fis4 d4 g4 
  fis4 fis4. e4 fis8   %5
  dis2 r4 e4 
  e4 d4 e8 dis8 e4~ 
  e4 dis4 b4 d4 
  d4 e4 d4 cis4 
  d2 r4 fis4   %10
  fis4 g4 g4 fis4 
  g2 r4 g4 
  g4 g4. fis8 g4 
  fis2 r4 fis4~ 
  fis8 e8 fis4. e4 dis8   %15
  b2.
  
}

tenor = \relative c' {
  \partial 4 b4 
    b8 a8 g8 a8 b4 c4 
  d2 r4 d4 
  e4 d4 c4 d4~ 
  d8 a8 d8 c8 b4 d4 
  d4 dis4 b4 c4   %5
  b2 r4 b4 
  c4 g4 c4 cis4 
  fis,8 b4 a8 g4 b4 
  b4 cis4. b8 a8 g8 
  fis2 r4 d'4   %10
  d4 d4 e4 d4 
  d2 r4 d4 
  e4 d4 e8 d8 d4 
  d2 r4 d8 c8 
  b2. b8. a16   %15
  g2.
  
}

baixo = \relative c {
  \partial 4 e4 
    b4 c4 b4 a4 
  g2 r4 g'8 fis8 
  e4 b4 c4 b8 c8 
  d2 g,4 g'4 
  d8 c8 b4 e4 a,4   %5
  b2 r4 e8 d8 
  c8 d8 c8 b8 a4 ais4 
  b2 e4 b8 cis8 
  d4 ais4 b4 e4 
  d2 r4 d4   %10
  d'8 c8 b4 a4 d,4 
  g2 r4 g8 fis8 
  e8 d8 c8 b8 a4 g4 
  d'2 r4 d4 
  dis8 cis8 dis8 b8 e4 b4   %15
  e,2. 
  
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