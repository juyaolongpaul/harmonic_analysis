\version "2.10.33"

\header {
  title = "344 - Vom Himmel hoch, da komm' ich her"
  composer = "J. S. Bach"
}


global = {
  \time 12/8
  \key g \major
}


soprano = \relative c''' {
  \partial 4. g4. 
  fis4. e4. fis4. d4. 
  e4. fis4. g4. g4. 
  g4. d4. d4 c8 b4 c8 
  d4. c4. b4. b4. 
  e4. e4. d4 e8 fis4.  %5
  g4 fis8 e4. d4. g4. 
  fis4. e4. d4. e4 d8 
  c4 b8 a4. g4. 

}

alto = \relative c'' {
  \partial 4. b4. 
  a4. b4 a8 a4. g4. 
  g4 c8 b4 a8 b4. b4. 
  b4. b4. a4. g4. 
  a4 e8 e8. g16 fis8 g4. g4. 
  g4. g4 a8 b4 a8 a4 b8  %5
  b4 a8 b4 a8 a4. b4. 
  c4 b8 b4 a8 a4 g8 g4. 
  g4. g4 fis8 d4.

}

tenor = \relative c' {
  \partial 4. d4. 
  d4. d4 cis8 d4 c8 c4 b8 
  b4 a8 d4. d4. d4. 
  e4. fis4 g8 d4. d4 e8 
  a,4 b8 c8. e16 dis8 e4. e4. 
  e4 d8 c4. b4 cis8 d4.  %5
  g,4 d'8 d4 cis8 fis4. e4. 
  e4 d8 d4 c8 c4 b8 c4 b8 
  c4 d8 e4 d8 b4. 

}

baixo = \relative c' {
  \partial 4. g4. 
  d'4. g,4 a8 d,4. g4. 
  c,4. d4. g,4. g'4 fis8 
  e4. b'4. fis4. g4 e8 
  fis4 gis8 a4. e4. e4 d8 
  c4 d8 e4 c8 g'4. fis4 b8  %5
  e,4 fis8 g4 a8 d,4. e4 g8 
  a4 b8 g4 a8 fis4 g8 c,4 d8 
  e4 d8 c4 d8 g,4. 

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